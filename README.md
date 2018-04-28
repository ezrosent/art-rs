# art-rs: Efficient ordered containers.
The adaptive radix tree (ART) is an efficient radix tree (aka trie) design introduced by [Leis, Kemper
and
Neumann](https://15721.courses.cs.cmu.edu/spring2018/papers/09-oltpindexes2/leis-icde2013.pdf) in 2013.
This includes an implementation of the ART data-structure described in that
work, along with experimental support for *prefix-caching*.

## Overview

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
  

## Prefix-Caching

Keys for this data-structure can be decomposed into byte sequences. Short byte
sequences can be hashed efficiently. This repo provides variants of the ART
that store a hash table mapping from key prefixes to *interior nodes* within
the tree. This allows traversals for either mutation operations or lookups to
skip several levels of the tree in their traversal.  This sort of trick is much
harder for ordered tree data-structures, as their keys do not necessarily have
the needed structure, and they may have more complicated rebalancing operations
which can make it more difficult to maintain the validity of the hash table.

The length of the cached prefixes can be customized, allowing you to limit the
maximum size of the cache.

## Performance

While not complete, we have a number of benchmarks that compare the ART-based
data-structures to rust's `HashSet` and `BTreeSet`. They show promising
performance for the vanilla ART implementation, and demonstrate that prefix
caching can improve performance even further when the set is large. See
`Performance.md` for more information and measurements.

## TODOs

This implementation if very rough, and likely contains bugs. On top of general
code improvements, there is also still a lot to do to get it to feature-parity
with the standard Rust container types.

### API Parity with `BTreeSet`
This includes good implementations of set operations, as well as a proper
iterator API. While we have a callback-based traversal API, we lack an idiomatic iterator
implementation 

### Bulk Insertions
The ART paper describes a method for performing optimized bulk insertions of
values, which is not yet implemented in this code-base.

### Multithreading
While [follow-up work](https://db.in.tum.de/~leis/papers/artsync.pdf)
implemented synchronization for the ART, this repo only includes a
single-threaded implementation. I am interested in implementing a multithreaded
version at some point in the future.

### Slab Allocation
We currently do heap allocations for all new interior nodes and leaf nodes. At
the very least interior nodes could probably benefit from slab allocation.

### Space-optimized Prefix Caching
Because real-world map workloads are often skewed towards a small subset of the
keys, it should be possible tune the prefix cache to store a small subset of
keys. This would reduce the space overhead of the cache while still hopefully
preserving most of the performance gains.


[1]: See section 4 of the paper for more information on this. In this code, it
is encapsulated by the `Digital` trait, which has implementation for common
integer and string types.
