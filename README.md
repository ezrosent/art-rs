# art-rs: Efficient ordered containers.
The adaptive radix tree (ART) is an efficient radix tree (aka trie) design introduced by [Leis, Kemper
and
Neumann](https://15721.courses.cs.cmu.edu/spring2018/papers/09-oltpindexes2/leis-icde2013.pdf) in 2013.
This includes an implementation of the ART data-structure described in that
work, along with experimental support for *prefix-caching*.

# Brief overview of Adaptive Radix Trees

ARTs operate on types that can be decomposed into sequences of bytes. If the
types are ordered, and these byte sequences (with a lexicographic ordering)
respect the ordering on the type, then ARTs also support efficient range scans.
Because most ordered types can also be efficiently decomposed to byte
sequences[1], this makes them a potential alternative to ordered tree
data-structures, like Rust's `BTreeSet` or `std::set` in C++ (often a Red-Black
Tree).

Compared to the classic [Trie datastructure](https://en.wikipedia.org/wiki/Trie),
the ART paper details lots of intricate optimizations to speed up lookups and
insertions. The most important of these are:

  * *Prefix Compression*: A sequence of interior that do not point directly to a
    leaf can be compressed into a single node, thus reducing the length of the
    path that must be traversed.

  * *Lazy Expansion*: A sequence of interior nodes that only point to a single
    leaf can be elided entirely.

  * *Specialized Interior Nodes*: Inner nodes in the tree have specialized
    implementations for ones with up to 4, 16, 48 and 256 children. This
    balances space efficiency with the speed of lookups.

See the ART paper for a more complete description of these features.
  

# Prefix-Caching

Keys for this data-structure can be decomposed into byte sequences. This repo
provides variants of the ART that store a hash table mapping from key prefixes
to *interior nodes* within the tree. This allows traversals for either mutation
operations or lookups to skip several levels of the tree in their traversal.
This sort of trick is much harder for ordered tree data-structures, as their
keys do not necessarily have the needed structure, and they may have more
complicated rebalancing operations which can make it more difficult to maintain
the validity of the hash table.

The length of the cached prefixes can be customized, allowing you to limit the
maximum size of the cache.

# Performance

We benchmarked lookups (keys within the set and keys not in the set) and
insert/delete pairs for `ARTSet` (our ART implementation), `CachingARTSet` (an
ART with a prefix cache), rust's `BTreeSet` and rust's `HashSet`. We use random
integer keys where the keys are chosen from 0 to the size of the set ("dense")
and where they are chosen from all possible 64-bit integers ("sparse"). We also
include benchmarks for random UTF-8 strings.


## Integers

Here we see that the ART generally does somewhere between the performance of the
BTree and the hash table. The cache is little help for small tables or dense
keys. This makes sense, as the sparse keys will often share a prefix, making the
likely depth of the tree fairly short. Prefix caching *does*, however, make a
substantial difference for sparse integers in larger tables.

(TODO link to graphs)

## Strings

There is a similar story here as to the integer workloads above. The benefit of
caching here is, however, more pronounced for both lookups and mutations.

(TODO link to graphs)


# TODOs

This implementation if very rough. There is still a lot to do to get it to
feature-parity with other Rust container types.

### API Parity with `BTReeSet`
This includes good implementations of set operations, as well as a proper
iterator API. While we have a callback-based traversal API, we lack an idiomatic iterator
implementation 

### Bulk Insertions
The ART paper describes a method for performing optimized bulk insertions of
values, which is not yet implemented in this code-base.

### Multithreading
While [follow-up work](https://db.in.tum.de/~leis/papers/artsync.pdf)
implemented synchronization for the ART, this repo only includes a single-threaded
implementation, though I am interested in implementing a multithreaded version
at some point in the future.

### Slab Allocation
We currently do heap allocations for all new interior nodes and leaf nodes. At
the very least interior nodes could probably benefit from slab allocation.


[1]: See section 4 of the paper for more information on this. In this code, it
is encapsulated by the `Digital` trait, which has implementation for common
integer and string types.
