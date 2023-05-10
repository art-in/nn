use network::network::Network;
use once_cell::sync::Lazy;

// large models fail in dev wasm build with stack overflow error while dropping lots
// of termporary BVals which are produced by network forward pass, so use small
// model for dev build and switch to large model in release mode only
#[cfg(not(debug_assertions))]
static MODEL: &[u8] =
    include_bytes!("../../train_runner/models/digits-784-1000-400-150-10/digits-784-1000-400-150-10-epoch-6-error-0.025.nm");

#[cfg(debug_assertions)]
static MODEL: &[u8] =
    include_bytes!("../../train_runner/models/digits-784-30-10/digits-784-30-10-epoch-4.nm");

thread_local!(pub static NETWORK: Lazy<Network> = Lazy::new(
    || Network::deserialize_from_reader(MODEL))
);

pub fn init_model() -> usize {
    // force model init earlier, so it doesn't slow down real first use
    NETWORK.with(|net| net.parameters().len())
}
