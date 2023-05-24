FROM rust:latest

# install rust components (nightly toolchain for dfdx lib)
RUN rustup toolchain install nightly-2022-10-01-aarch64-unknown-linux-gnu
RUN rustup component add rustfmt clippy --toolchain nightly-2022-10-01-aarch64-unknown-linux-gnu

# install nodejs (for web crates)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# install global utils (for web crates)
RUN cargo install wasm-pack@0.11.0
RUN npm install -g serve@14.2.0