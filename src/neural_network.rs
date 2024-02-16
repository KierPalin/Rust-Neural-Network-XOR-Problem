use crate::linear_algebra::*;
use crate::layers::*;
use crate::database::*;


/**
 * High level Artificial Neural Network abstraction
 */
pub struct NeuralNetwork {
    database: Database,
    layers: Vec<HiddenLayer>,
    epochs: usize,
    show_model_outputs: bool
}


impl NeuralNetwork {
    pub fn new(database: Database, layers: Vec<HiddenLayer>, epochs: usize, show_model_outputs: bool) -> NeuralNetwork {
        NeuralNetwork {
            database,
            layers,
            epochs,
            show_model_outputs
        }
    }
    
    /**
     * Exposed method used to generate the model to solve the dataset,
     * Will invoke .train() until the model is able to predict the dataset
     * 
     * This is neccessary for the XOR dataset, especially for low layer counts & sizes,
     *      because the model can get stuck in local minimas
     */
    pub fn generate_model(&mut self, minimum_accuracy: f32) {
        let mut current_round: usize = 1;

        while self.test() < minimum_accuracy {
            println!("\n\nTraining round: {}", current_round);
            current_round += 1;

            self.reset_model();
            self.train();
        }
        println!(" after {} rounds of training.", current_round)
    }

    /**
     * Internal method used to randomise the weights and biases for all layers again.
     */
    fn reset_model(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_weights_and_biases();
        }
    }

    /**
     * Unbatched training
     * Runs throughg the entire training dataset
     * Used by .generate_model()
     */
    fn train(&mut self) {
        for _ in 1..=self.epochs {
            for _ in 0..self.database.training_set_size {
                let (input, labels) = self.database.next_training();

                let network_out = self.forward(&input);
                let network_loss = NeuralNetwork::mse_loss_gradient(&network_out, &labels);

                self.backward(&network_loss);
            }
        }
    }

    /**
     * Calculates the accuracy of the model using the entire dataset (training and testing are identical in XOR)
     */
    fn test(&mut self) -> f32 {
        let mut correct_predictions = 0.0f32;

        for _ in 1..=self.database.training_set_size {
            let (input, labels) = self.database.next_testing();
            let classifications = self.classify(&input);
            
            if self.show_model_outputs {
                println!("Network input:\n{}", input);
                println!("Input labels:\n{}", labels);
                println!("Network classifications:\n{}\n", classifications);
            }

            if classifications == labels {
                correct_predictions += 1.0f32;
            }
        }
        
        if self.show_model_outputs {
            print!("Testing resulted in a model accuracy of: {}%", 
            (correct_predictions / self.database.training_set_size as f32) * 100.0f32);
        }
        correct_predictions / self.database.training_set_size as f32
    }


    /**
     * One-hot encode the output of the network,
     * so that it may be compared with the correct label outputs.
     */
    fn classify(&mut self, input: &Matrix) -> Matrix {
        Matrix::one_hot_encode_by_maximum(&self.forward(input))
    }

    /**
     * Forward propogation of granted input through each layer of the network.
     */
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }

        output
    }

    /**
     * Backward propogation of the gradient calculated from the forward propogation,
     * Invocation of each layer.backward to update each layers weights & Biases
     */
    fn backward(&mut self, downstream_grad: &Matrix) {
        let mut next_grad = downstream_grad.clone();
        
        for layer in self.layers.iter_mut().rev() {
            next_grad = layer.backward(&next_grad);
        }
    }

    /**
     * Calculate the loss gradient for Mean-Squared-Error
     */
    fn mse_loss_gradient(pred_matrix: &Matrix, true_matrix: &Matrix) -> Matrix {
        pred_matrix - true_matrix
    }
}