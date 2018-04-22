# ART Performance

We benchmarked lookups (keys within the set and keys not in the set) and
insert/delete pairs for `ARTSet` (our ART implementation), `CachingARTSet` (an
ART with a prefix cache), rust's `BTreeSet` and rust's `HashSet`. We use random
integer keys where the keys are chosen from 0 to the size of the set ("dense")
and where they are chosen from all possible 64-bit integers ("sparse"). We also
include benchmarks for random UTF-8 strings.

### Integers

Here we see that the ART generally does somewhere between the performance of
the BTree and the hash table. The cache is little help for small tables or
dense keys. This makes sense, as the dense keys will often share a prefix,
making the likely depth of the tree fairly short, while prefix compression will
ensure the absolute depth of the tree is quite low when there are few elements.
Prefix caching *does*, however, make a substantial difference for sparse
integers in larger tables.

![Integer Performance 16K](graphs/dense_u64_sparse_u64_lookup_miss_16384.png?raw=true)
![Integer Performance 16M](graphs/dense_u64_sparse_u64_lookup_miss_16777216.png?raw=true)
![Integer Performance 256M](graphs/dense_u64_sparse_u64_lookup_miss_268435456.png?raw=true)

![Integer Performance 16K](graphs/dense_u64_sparse_u64_lookup_hit_16384.png?raw=true)
![Integer Performance 16M](graphs/dense_u64_sparse_u64_lookup_hit_16777216.png?raw=true)
![Integer Performance 256M](graphs/dense_u64_sparse_u64_lookup_hit_268435456.png?raw=true)

![Integer Performance 16K](graphs/dense_u64_sparse_u64_insert_remove_16384.png?raw=true)
![Integer Performance 16M](graphs/dense_u64_sparse_u64_insert_remove_16777216.png?raw=true)
![Integer Performance 256M](graphs/dense_u64_sparse_u64_insert_remove_268435456.png?raw=true)

### Strings

There is a similar story here as to the integer workloads above. The benefit of
caching here is, however, more pronounced for both lookups and mutations.

![String Performance 16K](graphs/String_lookup_hit_16384.png?raw=true)
![String Performance 1M](graphs/String_lookup_hit_1048576.png?raw=true)
![String Performance 16M](graphs/String_lookup_hit_16777216.png?raw=true)

![String Performance 16K](graphs/String_lookup_miss_16384.png?raw=true)
![String Performance 1M](graphs/String_lookup_miss_1048576.png?raw=true)
![String Performance 16M](graphs/String_lookup_miss_16777216.png?raw=true)

![String Performance 16K](graphs/String_insert_remove_16384.png?raw=true)
![String Performance 1M](graphs/String_insert_remove_1048576.png?raw=true)
![String Performance 16M](graphs/String_insert_remove_16777216.png?raw=true)


