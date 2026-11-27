# Changelog

All notable changes to muninn are recorded here, written for people who use muninn rather than people who read its source.

## Unreleased

- Project started, under the MIT license.
- The foundation is in place: a shared file format with corruption detection, exact (brute-force) vector search with cosine, dot-product and Euclidean similarity, and filtering of results by document.
- Search results can be filtered by document fields: exact matches on labels (for example "language is en"), a list of allowed values ("tag is a or b"), number ranges ("year 2020 or later", "price below 10"), whether a field is present at all, and any combination of these with and, or and not. Filters can also be written as JSON, for example `{"and":[{"eq":{"lang":"en"}},{"range":{"year":{"gte":2020}}}]}`, and a mistake in a filter is reported with the exact place it occurs.
- Each collection declares which document fields can be filtered and what kind of value they hold: a label, a whole number, a decimal number, or true/false. Fields inside nested objects are reached with a dotted name such as `meta.lang`. A field holding a list matches when any item in the list matches. A document whose field holds the wrong kind of value (for example text where a number is expected) is simply treated as not having that field, rather than causing an error.
- Newly added documents can be filtered on immediately. Filter data is stored compactly on disk (about 45 bytes per document with six filterable fields in our tests), opens instantly, is checked for corruption, and is combined when parts of a collection are merged, with deleted documents removed.
- Filtering is fast: in our tests on 1 million documents, a single condition takes between 2 and 80 microseconds and a combination of three conditions at most about 0.2 milliseconds, for stored and newly added documents alike.
