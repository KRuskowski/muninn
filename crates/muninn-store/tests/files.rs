//! Immutable files: id maps, payloads, tombstones, manifests, plus the
//! directory lock, memory maps and atomic replacement.

mod common;

use std::collections::BTreeMap;
use std::io::Write;

use common::TestDir;
use muninn_core::container::{Container, tag};
use muninn_core::{AlignedBuf, BitSet, Error, Region};
use muninn_store::fs::{
  TEMP_SUFFIX, temp_path_for, write_atomic, write_atomic_with,
  write_container_file,
};
use muninn_store::{
  Advice, DirLock, ExternalId, ExternalIdRef, FrozenIdMap, IdMap,
  KEEP_MANIFESTS, Manifest, ManifestStore, PayloadBuilder, PayloadStore,
  SegmentEntry, latest_tombstones, manifest_file_name, map_file, map_file_with,
  read_tombstones, segment_file_name, tombstone_file_name, write_payloads,
  write_tombstones,
};

fn mixed_map(n: u64) -> IdMap {
  let mut m = IdMap::new();
  for i in 0..n {
    let id = if i % 3 == 0 {
      ExternalId::Str(format!("doc-{i}-ü"))
    } else {
      ExternalId::U64(i * 1_000_003)
    };
    let (doc, old) = m.insert(id).unwrap();
    assert_eq!((doc, old), (i as u32, None));
  }
  m
}

fn reopen(m: &IdMap) -> FrozenIdMap {
  let buf = m.write_to(AlignedBuf::new()).unwrap();
  FrozenIdMap::open(Region::new(buf)).unwrap()
}

#[test]
fn idmap_round_trip_mixed_ids() {
  let mut m = mixed_map(3000);
  // Update: re-inserting an id points it at a new doc.
  let (doc, old) = m.insert(ExternalId::U64(1_000_003)).unwrap();
  assert_eq!((doc, old), (3000, Some(1)));
  // Delete.
  assert_eq!(m.remove(ExternalIdRef::Str("doc-3-ü")), Some(3));
  assert_eq!(m.remove(ExternalIdRef::Str("doc-3-ü")), None);
  assert_eq!(m.len(), 3001);
  assert_eq!(m.live_len(), 2999);

  let f = reopen(&m);
  assert_eq!(f.len(), 3001);
  assert_eq!(f.live_len(), 2999);
  for i in 0..3000u64 {
    let (ext, expect) = if i % 3 == 0 {
      let s = format!("doc-{i}-ü");
      let e = (i != 3).then_some(i as u32);
      assert_eq!(f.get_str(&s), e);
      assert_eq!(m.get(ExternalIdRef::Str(&s)), e);
      (ExternalId::Str(s), e)
    } else {
      let k = i * 1_000_003;
      let e = Some(if i == 1 { 3000 } else { i as u32 });
      assert_eq!(f.get_u64(k), e);
      (ExternalId::U64(k), e)
    };
    assert_eq!(f.get(ext.as_borrowed()), expect);
    // Forward direction keeps deleted and superseded docs.
    assert_eq!(f.external_id(i as u32).unwrap(), ext.as_borrowed());
    assert_eq!(m.external_id(i as u32), Some(ext.as_borrowed()));
  }
  assert_eq!(f.external_id(3000).unwrap(), ExternalIdRef::U64(1_000_003));
  assert!(f.external_id(3001).is_err());
  assert_eq!(f.get_u64(7), None);
  assert_eq!(f.get_str("doc-1-ü"), None);
  assert_eq!(f.get_str(""), None);
  // Numeric 0 and string "0" are different ids.
  assert_eq!(f.get_str("0"), None);
}

#[test]
fn idmap_single_kind_maps_round_trip() {
  let mut nums = IdMap::new();
  let mut strs = IdMap::new();
  for i in 0..1000u64 {
    nums.insert(ExternalId::U64(u64::MAX - i)).unwrap();
    strs.insert(ExternalId::Str(format!("{i:x}"))).unwrap();
  }
  let (fnum, fstr) = (reopen(&nums), reopen(&strs));
  for i in 0..1000u64 {
    assert_eq!(fnum.get_u64(u64::MAX - i), Some(i as u32));
    assert_eq!(fstr.get_str(&format!("{i:x}")), Some(i as u32));
    assert_eq!(fnum.get_str(&format!("{i:x}")), None);
    assert_eq!(fstr.get_u64(i), None);
  }
  assert_eq!(fstr.external_id(10).unwrap(), ExternalIdRef::Str("a"));
}

#[test]
fn idmap_file_is_mapped_zero_copy() {
  let d = TestDir::new("idmap-file");
  let m = mixed_map(500);
  let path = d.path().join(segment_file_name(1, "idmap"));
  let len = m.write_file(&path).unwrap();
  assert_eq!(std::fs::metadata(&path).unwrap().len(), len);
  let region = map_file_with(&path, Advice::Random).unwrap();
  let base = region.bytes().as_ptr() as usize;
  Container::open(region.clone())
    .unwrap()
    .verify_checksum()
    .unwrap();
  let f = FrozenIdMap::open(region).unwrap();
  let ExternalIdRef::Str(s) = f.external_id(0).unwrap() else {
    panic!("doc 0 has a string id")
  };
  // The returned &str points into the mapped file.
  let p = s.as_ptr() as usize;
  assert!(p >= base && p < base + len as usize);
  assert_eq!(f.get_u64(499 * 1_000_003), Some(499));
}

#[test]
fn idmap_rejects_corrupt_files() {
  let buf = mixed_map(200).write_to(AlignedBuf::new()).unwrap();
  let bytes = buf.as_slice();
  // Truncation and wrong kind fail to open.
  assert!(
    FrozenIdMap::open(Region::copy_from(&bytes[..bytes.len() - 1])).is_err()
  );
  let pay = PayloadBuilder::new().write_to(AlignedBuf::new()).unwrap();
  assert!(FrozenIdMap::open(Region::new(pay)).is_err());
  // A flipped byte anywhere is rejected at open, or caught by the
  // checksum, or at worst makes lookups miss or report corruption; it
  // never panics or loops.
  for i in 0..bytes.len() {
    let mut bad = bytes.to_vec();
    bad[i] ^= 0x5A;
    let r = Region::copy_from(&bad);
    if let Ok(f) = FrozenIdMap::open(r) {
      for d in 0..f.len() as u32 {
        let _ = f.external_id(d);
      }
      for k in 0..200u64 {
        let _ = f.get_u64(k * 1_000_003);
        let _ = f.get_str(&format!("doc-{k}-ü"));
      }
    }
  }
}

#[test]
fn payload_round_trip_and_streaming_writer_agree() {
  let mut b = PayloadBuilder::new();
  let docs: Vec<Vec<u8>> = (0..1000)
    .map(|i| {
      if i % 10 == 0 {
        Vec::new()
      } else {
        format!(r#"{{"i":{i},"tag":"{}"}}"#, "x".repeat(i % 50)).into_bytes()
      }
    })
    .collect();
  for (i, d) in docs.iter().enumerate() {
    assert_eq!(b.push(d), i as u32);
  }
  assert_eq!(b.get(5), Some(docs[5].as_slice()));
  assert_eq!(b.get(1000), None);
  let built = b.write_to(Vec::new()).unwrap();
  let streamed =
    write_payloads(Vec::new(), docs.iter().map(Vec::as_slice)).unwrap();
  assert_eq!(built, streamed);

  let d = TestDir::new("payload");
  let path = d.path().join(segment_file_name(4, "payload"));
  b.write_file(&path).unwrap();
  let s = PayloadStore::open(map_file(&path).unwrap()).unwrap();
  assert_eq!(s.len(), 1000);
  for (i, doc) in docs.iter().enumerate() {
    assert_eq!(s.get(i as u32).unwrap(), doc.as_slice());
  }
  assert!(s.get(1000).is_err());
  let empty = PayloadStore::open(Region::new(
    PayloadBuilder::new().write_to(AlignedBuf::new()).unwrap(),
  ))
  .unwrap();
  assert!(empty.is_empty());
}

#[test]
fn payload_rejects_corrupt_files() {
  let mut b = PayloadBuilder::new();
  for i in 0..50u32 {
    b.push(format!("{{\"n\":{i}}}").as_bytes());
  }
  let buf = b.write_to(AlignedBuf::new()).unwrap();
  let bytes = buf.as_slice();
  assert!(
    PayloadStore::open(Region::copy_from(&bytes[..bytes.len() - 64])).is_err()
  );
  assert!(
    PayloadStore::open(Region::new(
      mixed_map(3).write_to(AlignedBuf::new()).unwrap()
    ))
    .is_err()
  );
  // Offsets section: make the last offset disagree with the data length.
  let whole = Region::new(buf.clone());
  let c = Container::open(whole.clone()).unwrap();
  let off = c.section(tag("offsets")).unwrap();
  let off_start =
    off.bytes().as_ptr() as usize - whole.bytes().as_ptr() as usize;
  let mut bad = bytes.to_vec();
  bad[off_start + 50 * 8] ^= 1;
  assert!(PayloadStore::open(Region::copy_from(&bad)).is_err());
  // A middle offset pointing past the data is caught on read.
  let mut bad = bytes.to_vec();
  bad[off_start + 10 * 8 + 6] = 0xFF;
  let s = PayloadStore::open(Region::copy_from(&bad)).unwrap();
  assert!(matches!(s.get(10), Err(Error::Corrupt(_))));
  assert!(s.get(20).is_ok());
}

#[test]
fn tombstones_round_trip_and_newest_version_wins() {
  let d = TestDir::new("tomb");
  let mut del = BitSet::new(1000);
  for i in (0..1000).step_by(7) {
    del.insert(i);
  }
  let n1 = write_tombstones(d.path(), 9, 1, &del).unwrap();
  assert_eq!(n1, tombstone_file_name(9, 1));
  let t = read_tombstones(&d.path().join(&n1)).unwrap();
  assert_eq!((t.segment_id, t.version), (9, 1));
  assert_eq!(t.deleted, del);

  del.insert(1);
  let n2 = write_tombstones(d.path(), 9, 2, &del).unwrap();
  write_tombstones(d.path(), 10, 5, &BitSet::new(3)).unwrap();
  let (name, t) = latest_tombstones(d.path(), 9).unwrap().unwrap();
  assert_eq!(name, n2);
  assert!(t.deleted.contains(1));
  assert_eq!(latest_tombstones(d.path(), 11).unwrap(), None);

  // A corrupt newest version falls back to the previous one.
  let p2 = d.path().join(&n2);
  let mut b = std::fs::read(&p2).unwrap();
  b[70] ^= 1;
  std::fs::write(&p2, &b).unwrap();
  assert!(read_tombstones(&p2).is_err());
  let (name, t) = latest_tombstones(d.path(), 9).unwrap().unwrap();
  assert_eq!(name, n1);
  assert!(!t.deleted.contains(1));
  // A truncated file is rejected too.
  std::fs::write(&p2, &b[..b.len() - 10]).unwrap();
  assert!(read_tombstones(&p2).is_err());
  // So is a file whose name lies about its version.
  std::fs::copy(d.path().join(&n1), d.path().join(tombstone_file_name(9, 3)))
    .unwrap();
  assert_eq!(latest_tombstones(d.path(), 9).unwrap().unwrap().0, n1);
}

fn manifest_with(segments: &[u64]) -> Manifest {
  let mut m = Manifest::new(serde_json::json!({
    "dim": 128, "metric": "cosine", "fields": ["title", "body"]
  }));
  for &id in segments {
    m.segments.push(SegmentEntry {
      id,
      doc_count: id * 10,
      files: BTreeMap::from([
        ("idmap".into(), segment_file_name(id, "idmap")),
        ("payload".into(), segment_file_name(id, "payload")),
      ]),
      tombstones: Some(tombstone_file_name(id, 1)),
      meta: serde_json::json!({"hnsw_m": 16}),
    });
    m.next_segment_id = m.next_segment_id.max(id + 1);
  }
  m
}

#[test]
fn manifest_commit_load_and_retention() {
  let d = TestDir::new("manifest");
  let (mut store, cur) = ManifestStore::open(d.path()).unwrap();
  assert!(cur.is_none());
  let mut m = manifest_with(&[1, 2]);
  m.wal_checkpoint_lsn = 77;
  store.commit(&mut m).unwrap();
  assert_eq!(m.generation, 1);
  let (_, loaded) = ManifestStore::open(d.path()).unwrap();
  assert_eq!(loaded.as_ref(), Some(&m));

  for g in 2..=5 {
    let mut next = manifest_with(&[g]);
    store.commit(&mut next).unwrap();
    assert_eq!(next.generation, g);
  }
  // Only the newest KEEP_MANIFESTS generations stay on disk.
  let manifests: Vec<_> = std::fs::read_dir(d.path())
    .unwrap()
    .filter_map(|e| e.unwrap().file_name().into_string().ok())
    .filter(|n| n.starts_with("MANIFEST-"))
    .collect();
  assert_eq!(manifests.len(), KEEP_MANIFESTS);
  let (store2, loaded) = ManifestStore::open(d.path()).unwrap();
  assert_eq!(loaded.unwrap().generation, 5);
  assert_eq!(store2.current().unwrap().segments[0].id, 5);
}

#[test]
fn manifest_survives_crash_between_temp_write_and_rename() {
  let d = TestDir::new("manifest-crash");
  let (mut store, _) = ManifestStore::open(d.path()).unwrap();
  let mut m = manifest_with(&[1]);
  store.commit(&mut m).unwrap();

  // Crash after writing (even fully) the next generation's temp file but
  // before the rename.
  let mut next = manifest_with(&[1, 2]);
  next.generation = 2;
  let target = d.path().join(manifest_file_name(2));
  let tmp = temp_path_for(&target);
  std::fs::write(&tmp, next.encode().unwrap()).unwrap();
  // And a half-written temp file from an earlier crash.
  let tmp2 = temp_path_for(&target);
  std::fs::write(&tmp2, &next.encode().unwrap()[..40]).unwrap();

  let (store, loaded) = ManifestStore::open(d.path()).unwrap();
  assert_eq!(loaded.unwrap(), m);
  let garbage = store.gc(true).unwrap();
  assert!(garbage.contains(&tmp) && garbage.contains(&tmp2));
  assert!(!tmp.exists() && !tmp2.exists());
}

#[test]
fn manifest_newest_corrupt_generation_falls_back() {
  let d = TestDir::new("manifest-corrupt");
  let (mut store, _) = ManifestStore::open(d.path()).unwrap();
  let mut a = manifest_with(&[1]);
  store.commit(&mut a).unwrap();
  let mut b = manifest_with(&[1, 2]);
  store.commit(&mut b).unwrap();
  let p = d.path().join(manifest_file_name(2));
  let good = std::fs::read(&p).unwrap();

  // Torn (truncated), bit-flipped, garbage, and wrong-generation copies of
  // the newest manifest all fall back to generation 1.
  let mut flipped = good.clone();
  flipped[100] ^= 0x01;
  // A change that keeps the JSON valid is caught by the checksum alone.
  let mut edited = good.clone();
  let at = good
    .windows(19)
    .position(|w| w == b"\"next_segment_id\": ")
    .unwrap()
    + 19;
  assert_eq!(edited[at], b'3');
  edited[at] = b'4';
  let mut gen3 = b.clone();
  gen3.generation = 3;
  let cases: Vec<Vec<u8>> = vec![
    good[..good.len() / 2].to_vec(),
    flipped,
    edited,
    b"{\"not\": \"a manifest\"}".to_vec(),
    Vec::new(),
    gen3.encode().unwrap(),
  ];
  for bytes in cases {
    std::fs::write(&p, &bytes).unwrap();
    let (mut s, loaded) = ManifestStore::open(d.path()).unwrap();
    assert_eq!(loaded.as_ref(), Some(&a));
    // A new commit goes above the corrupt generation, never reuses it.
    let mut c = manifest_with(&[3]);
    s.commit(&mut c).unwrap();
    assert_eq!(c.generation, 3);
    std::fs::remove_file(d.path().join(manifest_file_name(3))).unwrap();
  }
  std::fs::write(&p, &good).unwrap();
  assert_eq!(ManifestStore::open(d.path()).unwrap().1, Some(b));
}

#[test]
fn gc_removes_only_unreferenced_muninn_files() {
  let d = TestDir::new("gc");
  let touch = |name: &str| std::fs::write(d.path().join(name), b"x").unwrap();
  let (mut store, _) = ManifestStore::open(d.path()).unwrap();
  let mut m1 = manifest_with(&[1, 2]);
  store.commit(&mut m1).unwrap();
  let mut m2 = manifest_with(&[2, 3]);
  store.commit(&mut m2).unwrap();
  for id in 1..=3 {
    touch(&segment_file_name(id, "idmap"));
    touch(&segment_file_name(id, "payload"));
    touch(&tombstone_file_name(id, 1));
  }
  touch(&segment_file_name(4, "idmap")); // orphan from a failed flush
  touch(&tombstone_file_name(2, 0)); // superseded tombstones
  touch("seg-0000000003.idmap.123.0.tmp");
  touch("user-notes.txt");
  std::fs::create_dir(d.path().join("wal")).unwrap();
  let _lock = DirLock::acquire(d.path()).unwrap();

  let found = store.gc(false).unwrap();
  let names: Vec<String> = found
    .iter()
    .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
    .collect();
  assert_eq!(
    names,
    vec![
      "seg-0000000002.0000000000.tomb".to_string(),
      "seg-0000000003.idmap.123.0.tmp".to_string(),
      "seg-0000000004.idmap".to_string(),
    ]
  );
  // Dry run deleted nothing.
  assert!(found.iter().all(|p| p.exists()));
  // Segment 1 is only in the previous manifest, which is retained as a
  // fallback, so its files survive until the next commit.
  let mut m3 = manifest_with(&[3]);
  store.commit(&mut m3).unwrap();
  let found = store.gc(true).unwrap();
  assert!(found.iter().all(|p| !p.exists()));
  let left: Vec<String> = std::fs::read_dir(d.path())
    .unwrap()
    .map(|e| e.unwrap().file_name().into_string().unwrap())
    .collect();
  assert!(!left.iter().any(|n| n.starts_with("seg-0000000001")));
  assert!(left.contains(&segment_file_name(2, "idmap")));
  assert!(left.contains(&"user-notes.txt".to_string()));
  assert!(left.contains(&"LOCK".to_string()));
  assert!(left.contains(&"wal".to_string()));
  assert!(!left.iter().any(|n| n.ends_with(TEMP_SUFFIX)));
}

#[test]
fn lock_contention_returns_an_error() {
  let d = TestDir::new("lock");
  let first = DirLock::acquire(d.path()).unwrap();
  let err = DirLock::acquire(d.path()).unwrap_err();
  match &err {
    Error::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock),
    other => panic!("unexpected {other:?}"),
  }
  let msg = err.to_string();
  assert!(msg.contains("already open"), "{msg}");
  assert!(msg.contains(&std::process::id().to_string()), "{msg}");
  // Released on drop.
  drop(first);
  let again = DirLock::acquire(d.path()).unwrap();
  assert!(again.path().ends_with("LOCK"));
}

#[test]
fn lock_contention_across_threads() {
  let d = TestDir::new("lock-threads");
  let _held = DirLock::acquire(d.path()).unwrap();
  let p = d.path().to_path_buf();
  let r = std::thread::spawn(move || DirLock::acquire(&p).is_err())
    .join()
    .unwrap();
  assert!(r);
}

#[test]
fn map_file_handles_empty_files_and_outlives_unlink() {
  let d = TestDir::new("mmap");
  let empty = d.path().join("empty");
  std::fs::write(&empty, b"").unwrap();
  let r = map_file(&empty).unwrap();
  assert!(r.is_empty());
  assert!(map_file(&d.path().join("missing")).is_err());

  let p = d.path().join("data");
  let data: Vec<u8> = (0..10_000u32).map(|i| i as u8).collect();
  std::fs::write(&p, &data).unwrap();
  let r = map_file_with(&p, Advice::WillNeed).unwrap();
  let tail = r.slice(5000..10_000).unwrap();
  drop(r);
  std::fs::remove_file(&p).unwrap();
  // The slice keeps the map alive after the region and the file are gone.
  assert_eq!(tail.bytes(), &data[5000..]);
  muninn_store::advise(&tail, Advice::Sequential).unwrap();
}

#[test]
fn atomic_replace_keeps_old_contents_on_failure() {
  let d = TestDir::new("atomic");
  let p = d.path().join("file");
  write_atomic(&p, b"v1").unwrap();
  write_atomic(&p, b"version 2").unwrap();
  assert_eq!(std::fs::read(&p).unwrap(), b"version 2");
  let r = write_atomic_with(&p, |w| {
    w.write_all(b"partial")?;
    Err(Error::invalid("writer failed"))
  });
  assert!(r.is_err());
  assert_eq!(std::fs::read(&p).unwrap(), b"version 2");
  // No temporary file is left behind.
  let names: Vec<_> = std::fs::read_dir(d.path()).unwrap().collect();
  assert_eq!(names.len(), 1);

  let cp = d.path().join("c");
  let len = write_container_file(&cp, tag("test"), 2, |w| {
    w.section(tag("a"), b"hello")?;
    w.pod_section(tag("b"), &[1u32, 2, 3])
  })
  .unwrap();
  let region = map_file(&cp).unwrap();
  assert_eq!(region.len() as u64, len);
  let c = Container::open_kind(region, tag("test"), 2).unwrap();
  c.verify_checksum().unwrap();
  assert_eq!(
    c.section(tag("b")).unwrap().cast::<u32>().unwrap(),
    &[1, 2, 3]
  );
}
