use core::fmt::Debug;
use pasta_curves::arithmetic::FieldExt;
use pasta_curves::{pallas, vesta};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::transmute;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub const TEST_SEED: [u8; 16] = [42; 16];

// Question: Should the naming of `PallasVDF` and `VestaVDF` be reversed?

#[derive(Debug, Clone, Copy)]
pub enum EvalMode {
    LTRSequential,
    LTRAddChainSequential,
    RTLSequential,
    RTLParallel,
    RTLAddChainSequential,
    RTLAddChainParallel,
}

impl EvalMode {
    pub fn all() -> Vec<EvalMode> {
        vec![
            Self::LTRSequential,
            Self::LTRAddChainSequential,
            Self::RTLSequential,
            Self::RTLParallel,
            Self::RTLAddChainSequential,
            Self::RTLAddChainParallel,
        ]
    }
}

#[derive(Debug)]
struct Sq(Arc<UnsafeCell<Box<[[u64; 4]]>>>);
unsafe impl Send for Sq {}
unsafe impl Sync for Sq {}

/// Modulus is that of `Fq`, which is the base field of `Vesta` and scalar field of `Pallas`.
#[derive(Debug)]
pub struct PallasVDF {
    eval_mode: EvalMode,
}

impl MinRootVDF<pallas::Scalar> for PallasVDF {
    fn new_with_mode(eval_mode: EvalMode) -> Self {
        PallasVDF { eval_mode }
    }

    // To bench with this on 3970x:
    // RUSTFLAG="-C target-cpu=native -g" taskset -c 0,40 cargo bench
    fn eval(&mut self, x: State<pallas::Scalar>, t: u64) -> State<pallas::Scalar> {
        match self.eval_mode {
            EvalMode::LTRSequential
            | EvalMode::LTRAddChainSequential
            | EvalMode::RTLAddChainSequential
            | EvalMode::RTLSequential => self.simple_eval(x, t),
            EvalMode::RTLAddChainParallel => self.eval_rtl_addition_chain(x, t),
            EvalMode::RTLParallel => self.eval_rtl(x, t),
        }
    }

    fn element(n: u64) -> pallas::Scalar {
        pallas::Scalar::from(n)
    }

    /// Pallas' inverse_exponent is 5, so we can hardcode this.
    fn inverse_step(x: pallas::Scalar) -> pallas::Scalar {
        x.mul(&x.square().square())
    }

    fn forward_step(&mut self, x: pallas::Scalar) -> pallas::Scalar {
        match self.eval_mode {
            EvalMode::LTRSequential => self.forward_step_ltr_sequential(x),
            EvalMode::RTLSequential => self.forward_step_rtl_sequential(x),
            EvalMode::RTLAddChainSequential => self.forward_step_sequential_rtl_addition_chain(x),
            EvalMode::LTRAddChainSequential => self.forward_step_ltr_addition_chain(x),
            _ => unreachable!(),
        }
    }
}

impl PallasVDF {
    /// Number of bits in exponent.
    fn bit_count() -> usize {
        254
    }

    // To bench with this on 3970x:
    // RUSTFLAG="-C target-cpu=native -g" taskset -c 0,40 cargo bench
    fn eval_rtl(&mut self, x: State<pallas::Scalar>, t: u64) -> State<pallas::Scalar> {
        let bit_count = Self::bit_count();
        let squares1 = Arc::new(UnsafeCell::new(vec![[0u64; 4]; 254].into_boxed_slice()));
        let sq = Sq(squares1);
        let ready = Arc::new(AtomicUsize::new(1)); // Importantly, not zero.
        let ready_clone = Arc::clone(&ready);

        crossbeam::scope(|s| {
            s.spawn(|_| {
                let squares = unsafe {
                    transmute::<&mut [[u64; 4]], &mut [pallas::Scalar]>(slice::from_raw_parts_mut(
                        (*sq.0.get()).as_mut_ptr(),
                        bit_count,
                    ))
                };

                macro_rules! store {
                    ($index:ident, $val:ident) => {
                        squares[$index] = $val;
                        ready.store($index, Ordering::SeqCst)
                    };
                }

                for _ in 0..t {
                    while ready.load(Ordering::SeqCst) != 0 {}

                    let mut next_square = squares[0];

                    #[allow(clippy::needless_range_loop)]
                    for i in 0..Self::bit_count() {
                        if i > 0 {
                            next_square = next_square.square();
                        };

                        store!(i, next_square);
                    }
                }
            });
            (0..t).fold(x, |acc, _| self.round_with_squares(acc, &sq, &ready_clone))
        })
        .unwrap()
    }

    // To bench with this on 3970x:
    // RUSTFLAG="-C target-cpu=native -g" taskset -c 0,40 cargo bench
    fn eval_rtl_addition_chain(
        &mut self,
        x: State<pallas::Scalar>,
        t: u64,
    ) -> State<pallas::Scalar> {
        let bit_count = Self::bit_count();
        let squares1 = Arc::new(UnsafeCell::new(vec![[0u64; 4]; 254].into_boxed_slice()));
        let sq = Sq(squares1);
        let ready = Arc::new(AtomicUsize::new(1)); // Importantly, not zero.
        let ready_clone = Arc::clone(&ready);

        crossbeam::scope(|s| {
            s.spawn(|_| {
                let squares = unsafe {
                    transmute::<&mut [[u64; 4]], &mut [pallas::Scalar]>(slice::from_raw_parts_mut(
                        (*sq.0.get()).as_mut_ptr(),
                        bit_count,
                    ))
                };

                macro_rules! store {
                    ($index:ident, $val:ident) => {
                        squares[$index] = $val;
                        ready.store($index, Ordering::SeqCst)
                    };
                }

                for _ in 0..t {
                    while ready.load(Ordering::SeqCst) != 0 {}

                    let mut next_square = squares[0];

                    let first_section_bit_count = 128;

                    #[allow(clippy::needless_range_loop)]
                    for i in 0..first_section_bit_count {
                        if i > 0 {
                            next_square = next_square.square();
                        };

                        store!(i, next_square);
                    }

                    let mut k = first_section_bit_count;

                    next_square = {
                        let mut x = next_square;

                        x = x.mul(&x.square());
                        x.mul(&x.square().square().square().square())
                    };

                    for j in 1..=(8 * 15 + 1) {
                        next_square = next_square.square();

                        if j % 8 == 1 {
                            store!(k, next_square);
                            k += 1;
                        }
                    }
                }
            });
            (0..t).fold(x, |acc, _| self.round_with_squares(acc, &sq, &ready_clone))
        })
        .unwrap()
    }

    /// one round in the slow/forward direction.
    #[inline]
    fn round_with_squares(
        &mut self,
        x: State<pallas::Scalar>,
        squares: &Sq,
        ready: &Arc<AtomicUsize>,
    ) -> State<pallas::Scalar> {
        State {
            x: match self.eval_mode {
                EvalMode::RTLParallel => self.forward_step_with_squares_naive_rtl(
                    pallas::Scalar::add(&x.x, &x.y),
                    squares,
                    ready,
                ),
                EvalMode::RTLAddChainParallel => {
                    self.forward_step_with_squares(pallas::Scalar::add(&x.x, &x.y), squares, ready)
                }
                _ => panic!("fell through in y_with_squares"),
            },
            // Increment the round.
            y: pallas::Scalar::add(&x.x, &x.i),
            i: pallas::Scalar::add(&x.i, &pallas::Scalar::one()),
        }
    }

    #[inline]
    fn forward_step_with_squares(
        &mut self,
        x: pallas::Scalar,
        squares: &Sq,
        ready: &Arc<AtomicUsize>,
    ) -> pallas::Scalar {
        let sq = squares.0.get();
        unsafe { (**sq)[0] = transmute::<pallas::Scalar, [u64; 4]>(x) };

        ready.store(0, Ordering::SeqCst);

        let mut remaining = Self::exponent();
        let mut acc = pallas::Scalar::one();

        let bit_count = Self::bit_count();
        let first_section_bit_count = 128;
        let second_section_bit_count = bit_count - first_section_bit_count;
        let n = first_section_bit_count + (second_section_bit_count / 8) + 1;

        for next_index in 1..=n {
            let current_index = next_index - 1;
            let limb_index = current_index / 64;
            let limb = remaining[limb_index];

            let one = (limb & 1) == 1;
            let in_second_section = next_index > first_section_bit_count;

            if in_second_section || one {
                while ready.load(Ordering::SeqCst)
                    < if next_index > 1 {
                        current_index
                    } else {
                        next_index
                    }
                {}

                let squares =
                    unsafe { transmute::<&[[u64; 4]], &[pallas::Scalar]>(&**(squares.0.get())) };
                let elt = squares[current_index];
                acc = acc.mul(&elt);
            };

            remaining[limb_index] = limb >> 1;
        }
        acc
    }

    #[inline]
    fn forward_step_with_squares_naive_rtl(
        &mut self,
        x: pallas::Scalar,
        squares: &Sq,
        ready: &Arc<AtomicUsize>,
    ) -> pallas::Scalar {
        let sq = squares.0.get();
        unsafe { (**sq)[0] = transmute::<pallas::Scalar, [u64; 4]>(x) };

        ready.store(0, Ordering::SeqCst);

        let mut remaining = Self::exponent();
        let mut acc = pallas::Scalar::one();

        let bit_count = Self::bit_count();
        let first_section_bit_count = bit_count - 1;
        let second_section_bit_count = bit_count - first_section_bit_count;
        let n = first_section_bit_count + (second_section_bit_count / 8) + 1;

        for next_index in 1..=n {
            let current_index = next_index - 1;
            let limb_index = current_index / 64;
            let limb = remaining[limb_index];

            let one = (limb & 1) == 1;
            let in_second_section = next_index > first_section_bit_count;

            if in_second_section || one {
                while ready.load(Ordering::SeqCst)
                    < if next_index > 1 {
                        current_index
                    } else {
                        next_index
                    }
                {}

                let squares =
                    unsafe { transmute::<&[[u64; 4]], &[pallas::Scalar]>(&**(squares.0.get())) };
                let elt = squares[current_index];
                acc = acc.mul(&elt);
            };

            remaining[limb_index] = limb >> 1;
        }
        acc
    }

    fn forward_step_ltr_addition_chain(&mut self, x: pallas::Scalar) -> pallas::Scalar {
        let sqr = |x: pallas::Scalar, i: u32| (0..i).fold(x, |x, _| x.square());

        let mul = |x: pallas::Scalar, y| x.mul(y);
        let sqr_mul = |x, n, y: pallas::Scalar| y.mul(&sqr(x, n));

        let q1 = x;
        let q10 = sqr(q1, 1);
        let q11 = mul(q10, &q1);
        let q101 = mul(q10, &q11);
        let q110 = sqr(q11, 1);
        let q111 = mul(q110, &q1);
        let q1001 = mul(q111, &q10);
        let q1111 = mul(q1001, &q110);
        let qr2 = sqr_mul(q110, 3, q11);
        let qr4 = sqr_mul(qr2, 8, qr2);
        let qr8 = sqr_mul(qr4, 16, qr4);
        let qr16 = sqr_mul(qr8, 32, qr8);
        let qr32 = sqr_mul(qr16, 64, qr16);
        let qr32a = sqr_mul(qr32, 5, q1001);
        let qr32b = sqr_mul(qr32a, 8, q111);
        let qr32c = sqr_mul(qr32b, 4, q1);
        let qr32d = sqr_mul(qr32c, 2, qr4);
        let qr32e = sqr_mul(qr32d, 7, q11);
        let qr32f = sqr_mul(qr32e, 6, q1001);
        let qr32g = sqr_mul(qr32f, 3, q101);
        let qr32h = sqr_mul(qr32g, 7, q101);
        let qr32i = sqr_mul(qr32h, 7, q111);
        let qr32j = sqr_mul(qr32i, 4, q111);
        let qr32k = sqr_mul(qr32j, 5, q1001);
        let qr32l = sqr_mul(qr32k, 5, q101);
        let qr32m = sqr_mul(qr32l, 3, q11);
        let qr32n = sqr_mul(qr32m, 4, q101);
        let qr32o = sqr_mul(qr32n, 3, q101);
        let qr32p = sqr_mul(qr32o, 6, q1111);
        let qr32q = sqr_mul(qr32p, 4, q1001);
        let qr32r = sqr_mul(qr32q, 6, q101);
        let qr32s = sqr_mul(qr32r, 37, qr8);
        sqr_mul(qr32s, 2, q1)
    }

    // Sequential RTL square-and-multiply.
    fn forward_step_rtl_sequential(&mut self, x: pallas::Scalar) -> pallas::Scalar {
        (0..254)
            .scan(x, |state, _| {
                let ret = *state;
                *state = (*state).square();
                Some(ret)
            })
            .fold(
                (Self::exponent(), pallas::Scalar::one(), 0),
                |(mut remaining, acc, count), elt| {
                    let limb_index = count / 64;
                    let limb = remaining[limb_index];

                    let one = (limb & 1) == 1;
                    let acc = if one { acc.mul(&elt) } else { acc };
                    remaining[limb_index] = limb >> 1;

                    (remaining, acc, count + 1)
                },
            )
            .1
    }

    // Sequential RTL square-and-multiply with optimized addition chain.
    fn forward_step_sequential_rtl_addition_chain(&mut self, x: pallas::Scalar) -> pallas::Scalar {
        let first_section_bit_count = 128;
        let acc = pallas::Scalar::one();

        // First section is same as rtl without addition chain.
        let (_, acc, _, square_acc) = (0..first_section_bit_count)
            .scan(x, |state, _| {
                let ret = *state;
                *state = (*state).square();
                Some(ret)
            })
            .fold(
                (Self::exponent(), acc, 0, pallas::Scalar::zero()),
                |(mut remaining, acc, count, _previous_elt), elt| {
                    let limb_index = count / 64;
                    let limb = remaining[limb_index];

                    let one = (limb & 1) == 1;
                    let acc = if one { acc.mul(&elt) } else { acc };
                    remaining[limb_index] = limb >> 1;

                    (remaining, acc, count + 1, elt)
                },
            );

        let square_acc = square_acc.mul(&square_acc.square());
        let square_acc = square_acc.mul(&square_acc.square().square().square().square());

        (0..122)
            .scan(square_acc, |state, _| {
                *state = (*state).square();

                Some(*state)
            })
            .fold((acc, 1), |(acc, count), elt| {
                if count % 8 == 1 {
                    (acc.mul(&elt), count + 1)
                } else {
                    (acc, count + 1)
                }
            })
            .0
    }
}

/// Modulus is that of `Fp`, which is the base field of `Pallas and scalar field of Vesta.
#[derive(Debug)]
pub struct VestaVDF {}
impl MinRootVDF<vesta::Scalar> for VestaVDF {
    fn new_with_mode(_eval_mode: EvalMode) -> Self {
        VestaVDF {}
    }

    fn element(n: u64) -> vesta::Scalar {
        vesta::Scalar::from(n)
    }
    /// Vesta's inverse_exponent is 5, so we can hardcode this.
    fn inverse_step(x: vesta::Scalar) -> vesta::Scalar {
        x.mul(&x.square().square())
    }
    fn forward_step(&mut self, x: vesta::Scalar) -> vesta::Scalar {
        let sqr = |x: vesta::Scalar, i: u32| (0..i).fold(x, |x, _| x.square());

        let mul = |x: vesta::Scalar, y| x.mul(y);
        let sqr_mul = |x, n, y: vesta::Scalar| y.mul(&sqr(x, n));

        let p1 = x;
        let p10 = sqr(p1, 1);
        let p11 = mul(p10, &p1);
        let p101 = mul(p10, &p11);
        let p110 = sqr(p11, 1);
        let p111 = mul(p110, &p1);
        let p1001 = mul(p111, &p10);
        let p1111 = mul(p1001, &p110);
        let pr2 = sqr_mul(p110, 3, p11);
        let pr4 = sqr_mul(pr2, 8, pr2);
        let pr8 = sqr_mul(pr4, 16, pr4);
        let pr16 = sqr_mul(pr8, 32, pr8);
        let pr32 = sqr_mul(pr16, 64, pr16);
        let pr32a = sqr_mul(pr32, 5, p1001);
        let pr32b = sqr_mul(pr32a, 8, p111);
        let pr32c = sqr_mul(pr32b, 4, p1);
        let pr32d = sqr_mul(pr32c, 2, pr4);
        let pr32e = sqr_mul(pr32d, 7, p11);
        let pr32f = sqr_mul(pr32e, 6, p1001);
        let pr32g = sqr_mul(pr32f, 3, p101);
        let pr32h = sqr_mul(pr32g, 5, p1);
        let pr32i = sqr_mul(pr32h, 7, p101);
        let pr32j = sqr_mul(pr32i, 4, p11);
        let pr32k = sqr_mul(pr32j, 8, p111);
        let pr32l = sqr_mul(pr32k, 4, p1);
        let pr32m = sqr_mul(pr32l, 4, p111);
        let pr32n = sqr_mul(pr32m, 9, p1111);
        let pr32o = sqr_mul(pr32n, 8, p1111);
        let pr32p = sqr_mul(pr32o, 6, p1111);
        let pr32q = sqr_mul(pr32p, 2, p11);
        let pr32r = sqr_mul(pr32q, 34, pr8);
        sqr_mul(pr32r, 2, p1)
    }
}

// Question: Is this right, or is it the reverse? Which scalar fields' modulus do we want to target?
pub type TargetVDF<'a> = PallasVDF;

#[derive(std::cmp::PartialEq, Debug, Clone, Copy)]
pub struct State<T> {
    pub x: T,
    pub y: T,
    pub i: T,
}

pub trait MinRootVDF<F>: Debug
where
    F: FieldExt,
{
    fn new() -> Self
    where
        Self: Sized,
    {
        Self::new_with_mode(Self::default_mode())
    }

    fn new_with_mode(eval_mode: EvalMode) -> Self;

    fn default_mode() -> EvalMode {
        EvalMode::LTRSequential
    }

    #[inline]
    /// Exponent used to take a root in the 'slow' direction.
    fn exponent() -> [u64; 4] {
        F::RESCUE_INVALPHA
    }

    #[inline]
    /// Exponent used in the 'fast' direction.
    fn inverse_exponent() -> u64 {
        F::RESCUE_ALPHA
    }

    #[inline]
    /// The building block of a round in the slow, 'forward' direction.
    fn forward_step_ltr_sequential(&mut self, x: F) -> F {
        x.pow_vartime(Self::exponent())
    }

    #[inline]
    /// The building block of a round in the slow, 'forward' direction.
    fn forward_step(&mut self, x: F) -> F {
        self.forward_step_ltr_sequential(x)
    }

    #[inline]
    /// The building block of a round in the fast, 'inverse' direction.
    fn inverse_step(x: F) -> F {
        x.pow_vartime([Self::inverse_exponent(), 0, 0, 0])
    }

    /// one round in the slow/forward direction.
    fn round(&mut self, s: State<F>) -> State<F> {
        State {
            x: self.forward_step(F::add(s.x, s.y)),
            y: F::add(s.x, s.i),
            i: F::add(s.i, F::one()),
        }
    }

    /// One round in the fast/inverse direction.
    fn inverse_round(s: State<F>) -> State<F> {
        let i = F::sub(s.i, &F::one());
        let x = F::sub(s.y, &i);
        let mut y = Self::inverse_step(s.x);
        y.sub_assign(&x);
        State { x, y, i }
    }

    /// Evaluate input `x` with time/difficulty parameter, `t` in the
    /// slow/forward direction.
    fn eval(&mut self, x: State<F>, t: u64) -> State<F> {
        self.simple_eval(x, t)
    }

    fn simple_eval(&mut self, x: State<F>, t: u64) -> State<F> {
        (0..t).fold(x, |acc, _| self.round(acc))
    }

    /// Invert evaluation of output `x` with time/difficulty parameter, `t` in
    /// the fast/inverse direction.
    fn inverse_eval(x: State<F>, t: u64) -> State<F> {
        (0..t).fold(x, |acc, _| Self::inverse_round(acc))
    }

    /// Quickly check that `result` is the result of having slowly evaluated
    /// `original` with time/difficulty parameter `t`.
    fn check(result: State<F>, t: u64, original: State<F>) -> bool {
        original == Self::inverse_eval(result, t)
    }

    fn element(n: u64) -> F;
}

#[derive(Debug)]
pub struct VanillaVDFProof<V: MinRootVDF<F> + Debug, F: FieldExt> {
    result: State<F>,
    t: u64,
    _v: PhantomData<V>,
}

impl<V: MinRootVDF<F>, F: FieldExt> VanillaVDFProof<V, F> {
    pub fn eval_and_prove(x: State<F>, t: u64) -> Self {
        let mut vdf = V::new();
        let result = vdf.eval(x, t);
        Self {
            result,
            t,
            _v: PhantomData::<V>,
        }
    }

    pub fn eval_and_prove_with_mode(eval_mode: EvalMode, x: State<F>, t: u64) -> Self {
        let mut vdf = V::new_with_mode(eval_mode);
        let result = vdf.eval(x, t);
        Self {
            result,
            t,
            _v: PhantomData::<V>,
        }
    }

    pub fn result(&self) -> State<F> {
        self.result
    }

    pub fn verify(&self, original: State<F>) -> bool {
        V::check(self.result, self.t, original)
    }

    pub fn append(&self, other: Self) -> Option<Self> {
        if other.verify(self.result) {
            Some(Self {
                result: other.result,
                t: self.t + other.t,
                _v: PhantomData::<V>,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_exponents() {
        test_exponents_aux::<PallasVDF, pallas::Scalar>();
        test_exponents_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_exponents_aux<V: MinRootVDF<F>, F: FieldExt>() {
        assert_eq!(V::inverse_exponent(), 5);
        assert_eq!(V::inverse_exponent(), 5);
    }

    #[test]
    fn test_steps() {
        test_steps_aux::<PallasVDF, pallas::Scalar>();
        test_steps_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_steps_aux<V: MinRootVDF<F>, F: FieldExt>() {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);
        let mut vdf = V::new();

        for _ in 0..100 {
            let x = F::random(&mut rng);
            let y = vdf.forward_step(x);
            let z = V::inverse_step(y);

            assert_eq!(x, z);
        }
    }

    #[test]
    fn test_eval() {
        println!("top");
        test_eval_aux::<PallasVDF, pallas::Scalar>();
    }

    fn test_eval_aux<V: MinRootVDF<F>, F: FieldExt>() {
        for mode in EvalMode::all().iter() {
            test_eval_aux2::<V, F>(*mode)
        }
    }

    fn test_eval_aux2<V: MinRootVDF<F>, F: FieldExt>(eval_mode: EvalMode) {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);
        let mut vdf = V::new_with_mode(eval_mode);

        for _ in 0..10 {
            let t = 10;
            let x = F::random(&mut rng);
            let y = F::random(&mut rng);
            let x = State { x, y, i: F::zero() };
            let result = vdf.eval(x, t);
            let again = V::inverse_eval(result, t);

            assert_eq!(x, again);
            assert!(V::check(result, t, x));
        }
    }

    #[test]
    fn test_vanilla_proof() {
        test_vanilla_proof_aux::<PallasVDF, pallas::Scalar>();
        test_vanilla_proof_aux::<VestaVDF, vesta::Scalar>();
    }

    fn test_vanilla_proof_aux<V: MinRootVDF<F>, F: FieldExt>() {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        let x = F::random(&mut rng);
        let y = F::zero();
        let x = State { x, y, i: F::zero() };
        let t = 12;
        let n = 11;

        let first_proof = VanillaVDFProof::<V, F>::eval_and_prove(x, t);

        let final_proof = (1..11).fold(first_proof, |acc, _| {
            let new_proof = VanillaVDFProof::<V, F>::eval_and_prove(acc.result, t);
            acc.append(new_proof).expect("failed to append proof")
        });

        assert_eq!(V::element(final_proof.t), final_proof.result.i);
        assert_eq!(n * t, final_proof.t);
        assert!(final_proof.verify(x));
    }
}
