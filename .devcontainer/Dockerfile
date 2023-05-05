FROM rust:latest

# install rust components
RUN rustup component add rustfmt clippy

# install nodejs
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# install global utils
RUN cargo install wasm-pack@0.11.0
RUN npm install -g serve@14.2.0