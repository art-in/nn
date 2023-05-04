FROM rust:latest

# install rust components
RUN rustup component add rustfmt clippy

# install nodejs
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# install wasm-pack
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# install serve
RUN npm install -g serve@14.2.0