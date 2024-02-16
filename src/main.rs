use xor_problem::neural_network::*;
use xor_problem::activation_functions::*;
use xor_problem::layers::HiddenLayer;
use xor_problem::database::Database;

fn main() {
    let database = Database::from_xor();
    
    // let layers = vec![
    //                 HiddenLayer::new(3, 2, sigmoid, d_sigmoid), 
    //                 HiddenLayer::new(2, 3, sigmoid, d_sigmoid)
    //             ];

    let layers = vec![
        HiddenLayer::new(3, 2, relu, d_relu), 
        HiddenLayer::new(3, 3, relu, d_relu), 
        HiddenLayer::new(2, 3, sigmoid, d_sigmoid)
    ];

    // let layers = vec![
    //                 HiddenLayer::new(2, 2, sigmoid, d_sigmoid),
    //                 HiddenLayer::new(2, 2, sigmoid, d_sigmoid)
    //             ];
    
    let mut ann = NeuralNetwork::new(database, layers, 1000, true);

    let desired_accuracy = 0.98f32;
    ann.generate_model(desired_accuracy);
}
