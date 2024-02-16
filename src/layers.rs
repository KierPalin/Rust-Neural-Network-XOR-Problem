use crate::linear_algebra::Matrix;
use crate::activation_functions::{ActivationFun, ActivationFunDerivative};

const LEARNING_RATE: f32 = 1.02f32;

pub struct HiddenLayer {
    weights: Matrix,
    biases: Matrix,
    input_cache: Option<Matrix>,
    hypothesis_cache: Option<Matrix>,
    activation: ActivationFun,
    activation_derivative: ActivationFunDerivative,
}


impl HiddenLayer {
    pub fn new(rows: usize, cols: usize, activation: ActivationFun, activation_derivative: ActivationFunDerivative) -> HiddenLayer {
        HiddenLayer {
            weights: Matrix::from_random(rows, cols),
            biases: Matrix::from_random(rows, 1),
            input_cache: None,
            hypothesis_cache: None,
            activation,
            activation_derivative,
        }
    }

    pub fn reset_weights_and_biases(&mut self) {
        self.weights = Matrix::from_random(self.weights.rows, self.weights.cols);
        self.biases = Matrix::from_random(self.biases.rows, 1);
    }


    //------------------------
    // Neural Network Methods:
    //------------------------

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input_cache = Some(input.clone());
        self.hypothesis_cache = Some(&(self.weights.dot_product(&input)) + &self.biases);
        (self.activation)(self.hypothesis_cache.as_ref().unwrap())
    }

    pub fn backward(&mut self, downstream_grad: &Matrix) -> Matrix {
        // Use the derivative of the activation on the cached hypothesis, apply this to incoming downstream:
        // Resulting in Partial Derivative of downstream wrto Activation
        let grad_wrto_hypothesis = downstream_grad * &((self.activation_derivative)(self.hypothesis_cache.as_ref().unwrap()));
        
        // Partial Derivative of Activation wrto weights, by applying inputs:
        let w_delta: &Matrix = &grad_wrto_hypothesis.dot_product(&self.input_cache.as_ref().unwrap().transpose());
        
        // Take columnwise sum, fill each element with the sum of its column, discount by rows
        let b_delta = &Matrix::apply_scalar(&Matrix::colwise_sum_maintain_dim(&grad_wrto_hypothesis), 1.0f32 / grad_wrto_hypothesis.rows as f32);
        
        // Gradient wrto so that the prior layer may recurse using this same method:
        // Partial Derivative of Input Wrto Activation, by applying weights:
        let next_downstream = self.weights.transpose().dot_product(&grad_wrto_hypothesis);

        // Discount the weights & biases to adjust for their responsibility for the output loss:
        self.weights -= Matrix::apply_scalar(w_delta, LEARNING_RATE);
        self.biases -= Matrix::apply_scalar(b_delta, LEARNING_RATE);

        next_downstream
    }
}