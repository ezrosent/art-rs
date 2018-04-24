use std::iter::Iterator;
use std::str;
use super::byteorder::{BigEndian, ByteOrder};

/// `Digital` describes types that can be expressed as sequences of bytes.
///
/// The type's `digits` should respect equality and ordering on the type.
/// Furthermore, if the `digits` of one value are a prefix of the `digits`
/// of another value of the same type, the two values must be equal.
///
/// TODO implement floating point support. This is described in the ART paper but a couple details
/// are left out.
///
/// TODO implement macro/derive that will create a "digits" representation for any ordered type.
pub trait Digital<'a> {
    // TODO: consider providing a more efficient interface here (e.g. passing a slice directly)
    type I: Iterator<Item = u8> + 'a;
    const STOP_CHARACTER: Option<u8> = None;
    fn digits(&'a self) -> Self::I;
}

pub struct U32BytesIterator {
    cursor: usize,
    bytes: [u8; 4],
}

impl Iterator for U32BytesIterator {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        if self.cursor < 4 {
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

impl<'a> Digital<'a> for u32 {
    type I = U32BytesIterator;
    fn digits(&self) -> U32BytesIterator {
        let mut res = U32BytesIterator {
            cursor: 0,
            bytes: [0; 4],
        };
        BigEndian::write_u32(&mut res.bytes, *self);
        res
    }
}

impl<'a> Digital<'a> for i32 {
    type I = U32BytesIterator;
    fn digits(&self) -> U32BytesIterator {
        let mut res = U32BytesIterator {
            cursor: 0,
            bytes: [0; 4],
        };
        BigEndian::write_i32(&mut res.bytes, *self ^ (1 << 31));
        res
    }
}

impl<'a> Digital<'a> for i64 {
    type I = U64BytesIterator;
    fn digits(&self) -> U64BytesIterator {
        let mut res = U64BytesIterator {
            cursor: 0,
            bytes: [0; 8],
        };
        BigEndian::write_i64(&mut res.bytes, *self ^ (1 << 63));
        res
    }
}

impl<'a> Digital<'a> for usize {
    // Just treat usize as u64. This should (inefficiently) support platforms with a smaller type,
    // and we debug-assert that usize <= u64 in size.
    type I = U64BytesIterator;
    fn digits(&self) -> Self::I {
        debug_assert!(::std::mem::size_of::<usize>() <= ::std::mem::size_of::<u64>());
        (*self as u64).digits()
    }
}

impl<'a> Digital<'a> for isize {
    type I = U64BytesIterator;
    fn digits(&self) -> Self::I {
        debug_assert!(::std::mem::size_of::<isize>() <= ::std::mem::size_of::<i64>());
        (*self as i64).digits()
    }
}

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
    const STOP_CHARACTER: Option<u8> = Some(0);
    fn digits(&'a self) -> Self::I {
        NullTerminate::new(self.bytes())
    }
}

impl<'a> Digital<'a> for String {
    type I = NullTerminate<str::Bytes<'a>>;
    const STOP_CHARACTER: Option<u8> = Some(0);
    fn digits(&'a self) -> Self::I {
        NullTerminate::new(self.as_str().bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_digits_obey_order<D: for<'a> Digital<'a> + PartialOrd>(x: D, y: D) -> bool {
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

        fn digits_u64(x: u64, y: u64) -> bool {
            // why shift left? the RNG seems to generate numbers <256, so endianness bugs do not
            // get caught!
            test_digits_obey_order(x.wrapping_shl(20), y.wrapping_shl(20))
        }

        fn digits_u32(x: u32, y: u32) -> bool {
            test_digits_obey_order(x.wrapping_shl(20), y.wrapping_shl(20))
        }

        fn digits_i32(x: i32, y: i32) -> bool {
            test_digits_obey_order(x.wrapping_mul(1 << 10), y.wrapping_mul(1 << 10))
        }

        fn digits_i64(x: i64, y: i64) -> bool {
            test_digits_obey_order(x.wrapping_mul(1 << 20), y.wrapping_mul(1 << 20))
        }

        fn digits_isize(x: isize, y: isize) -> bool {
            test_digits_obey_order(x.wrapping_mul(1 << 20), y.wrapping_mul(1 << 20))
        }

        fn digits_usize(x: usize, y: usize) -> bool {
            test_digits_obey_order(x.wrapping_shl(20), y.wrapping_shl(20))
        }
    }
}
