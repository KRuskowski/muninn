//! The `muninn` command-line tool; see `muninn --help`.

fn main() -> std::process::ExitCode {
  muninn_server::cli::main()
}
