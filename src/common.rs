use std::iter::Iterator;
use std::str;
use super::byteorder::{BigEndian, ByteOrder};

/// `Digital` describes types that can be expressed as sequences of bytes.
///
/// The type's `digits` should respect equality and ordering on the type.
/// Furthermore, if the `digits` of one value are a prefix of the `digits`
/// of another value of the same type, the two values must be equal.
pub trait Digital<'a> {
    type I: Iterator<Item = u8> + 'a;
    fn digits(&'a self) -> Self::I;
}

// TODO: Add implementation for the rest of the numeric, string types
pub struct U64BytesIterator {
    cursor: usize,
    bytes: [u8; 8],
}

impl Iterator for U64BytesIterator {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.cursor < 8 {
            self.cursor += 1;
            Some(self.bytes[self.cursor - 1])
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<u8> {
        self.cursor += n;
        self.next()
    }
}

impl<'a> Digital<'a> for u64 {
    type I = U64BytesIterator;
    fn digits(&self) -> U64BytesIterator {
        let mut res = U64BytesIterator {
            cursor: 0,
            bytes: [0; 8],
        };
        BigEndian::write_u64(&mut res.bytes, *self);
        res
    }
}

/// NullTerminate transforms iterator corresponding to the bytes of a valid UTF-8 string into an
/// iterator suitable for use in a `Digital` implementation. This comes for free in languages using
/// C-style ASCII strings by convention, because null-termination guarantees the "prefixes"
/// property of the trait.
///
/// In Rust, strings are most commonly encoded as UTF-8. For such strings,  NUL characters are
/// kosher in the middle of a string, and picking a different byte as a terminator character will
/// ruin the compatibility with Ord[0]. To ensure that a null terminator is valid, we increase the
/// value of all bytes emitted by `I` by 1. We are guaranteed no overflow by the fact that 255 is
/// an invalid byte for UTF-8 strings. Given no overflow, equality and ordering are clearly
/// conserved.
///
/// [0]: To see why this is the case, consider the example of "" and "a". "" < "a", but "\u{255}" >
/// "a\u{255}".
pub struct NullTerminate<I> {
    done: bool,
    i: I,
}

impl<I> NullTerminate<I> {
    fn new(i: I) -> Self {
        NullTerminate { done: false, i: i }
    }
}

impl<I: Iterator<Item = u8>> Iterator for NullTerminate<I> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.done {
            return None;
        }
        let res = self.i.next();
        if let Some(s) = res {
            debug_assert!(s < 255);
            Some(s + 1)
        } else {
            self.done = true;
            Some(0)
        }
    }

    fn nth(&mut self, n: usize) -> Option<u8> {
        if self.done {
            return None;
        }
        let (remaining, _max) = self.i.size_hint();
        debug_assert_eq!(
            Some(remaining),
            _max,
            "must use iterator with exact length for NullTerminate"
        );
        if n + 1 == remaining {
            self.done = true;
            Some(0)
        } else {
            self.i.nth(n).map(|x| x + 1)
        }
    }
}

impl<'a> Digital<'a> for str {
    type I = NullTerminate<str::Bytes<'a>>;
    fn digits(&'a self) -> Self::I {
        NullTerminate::new(self.bytes())
    }
}

impl<'a> Digital<'a> for String {
    type I = NullTerminate<str::Bytes<'a>>;
    fn digits(&'a self) -> Self::I {
        NullTerminate::new(self.as_str().bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_digits_obey_order<D: for<'a> Digital<'a> + Ord>(x: D, y: D) -> bool {
        let vx: Vec<_> = x.digits().collect();
        let vy: Vec<_> = y.digits().collect();
        if x < y {
            vx < vy
        } else {
            vx >= vy
        }
    }

    quickcheck! {
        fn digits_strings(x: String, y: String) -> bool {
            test_digits_obey_order(x, y)
        }

        fn digits_unsigned_ints(x: u64, y: u64) -> bool {
            // why shift left? the RNG seems to generate numbers <256, so endianness bugs do not
            // get caught!
            test_digits_obey_order(x.wrapping_shl(20), y.wrapping_shl(20))
        }
    }
}
