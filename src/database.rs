use crate::linear_algebra::Matrix;

pub struct Database {
    pub training_set_size: usize,
    pub testing_set_size: usize,

    training_data: Vec<Matrix>,
    training_labels: Vec<Matrix>,
    current_training_index: usize,
    testing_data: Vec<Matrix>,
    testing_labels: Vec<Matrix>,
    current_testing_index: usize,
}

impl Database {
    /**
     * The training and testing datasets will be the same for XOR
     * XOR inputs are -1 & 1; as per Digital Image Processing 4th Edition,
     * Graphical exploration of the issue is simpler with these inputs,
     *      whilst the relationship between the issues remains the same.
     */
    pub fn from_xor() -> Database {
        let training_data = vec![
                                Matrix::from_2d_vec(2, 1, vec![vec![1.0f32], vec![1.0f32]]), 
                                Matrix::from_2d_vec(2, 1, vec![vec![-1.0f32], vec![-1.0f32]]),
                                Matrix::from_2d_vec(2, 1, vec![vec![-1.0f32], vec![1.0f32]]),
                                Matrix::from_2d_vec(2, 1, vec![vec![1.0f32], vec![-1.0f32]])
                            ];

        let testing_data = training_data.clone();

        // One-hot encoding of correct response:
        let training_labels = vec![
                                Matrix::from_2d_vec(2, 1, vec![vec![1.0f32], vec![0.0f32]]), 
                                Matrix::from_2d_vec(2, 1, vec![vec![1.0f32], vec![0.0f32]]),
                                Matrix::from_2d_vec(2, 1, vec![vec![0.0f32], vec![1.0f32]]),
                                Matrix::from_2d_vec(2, 1, vec![vec![0.0f32], vec![1.0f32]])
                            ];

        let testing_labels = training_labels.clone();
        
        Database {
            training_set_size: 4,
            testing_set_size: 4,

            training_data,
            training_labels,
            current_training_index: 0,
            
            testing_data,
            testing_labels,
            current_testing_index: 0
        }
    }

    /**
     * Grants tuple of (training_data, training_labels),
     * Iterates the current_index
     * Will wrap around data set
     */
    pub fn next_training(&mut self) -> (Matrix, Matrix) {
        let next: (Matrix, Matrix) = 
                    (self.training_data[self.current_training_index % self.training_set_size].clone(), 
                    self.training_labels[self.current_training_index % self.training_set_size].clone());
        self.current_training_index += 1;
        next
    }

    /**
     * Grants tuple of (testing_data, testing_labels),
     * Iterates the current_index
     * Will wrap around data set
     */
    pub fn next_testing(&mut self) -> (Matrix, Matrix) {
        let next: (Matrix, Matrix) = 
                    (self.testing_data[self.current_testing_index % self.testing_set_size].clone(), 
                    self.testing_labels[self.current_testing_index % self.testing_set_size].clone());
        self.current_testing_index += 1;
        next
    }
}