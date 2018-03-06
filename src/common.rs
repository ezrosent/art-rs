extern crate byteorder;

use std::iter::Iterator;
use std::str;
use self::byteorder::{BigEndian, ByteOrder};
 
/// `Digital` describes types that can be expressed as sequences of bytes.
///
/// The type's `digits` should respect equality and ordering on the type.
pub trait Digital<'a> {
    type I: Iterator<Item = u8> + 'a;
    fn digits(&'a self) -> Self::I;
}

// TODO: Add implementation for the rest of the numeric, string types
// TODO: add implementations of `nth` here to speed up use of `skip` in ART implementation

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
        if res.is_none() {
            self.done = true;
            Some(0)
        } else {
            res
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
