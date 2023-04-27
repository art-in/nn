use autograd::network::Network;

pub fn get_network_model_file_name(prefix: &str, net: &Network) -> String {
    let mut res = prefix.to_string();

    res += "-";

    let input_size = net.layers[0].neurons[0].weights.len();
    res += &input_size.to_string();

    for layer in &net.layers {
        res += "-";
        res += &layer.neurons.len().to_string();
    }

    res += ".nm";

    res
}
