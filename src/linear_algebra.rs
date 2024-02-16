use std::ops::{Index, IndexMut, Add, Sub, Mul, Div, SubAssign};
use std::fmt::{Result, Display, Formatter};

use rand_distr::{Normal, Distribution};

const GAUSSIAN_MEAN: f32 = 0.0f32;
const GAUSSIAN_STD_DEVIATION: f32 = 0.02f32;


#[derive(Clone)]
pub struct Matrix {
    data: Vec<Vec<f32>>,
    pub rows: usize, 
    pub cols: usize
}

impl Matrix {
    /**
     * Inefficient random value generation
     * Generated in accordance with DEFAULT_GAUSSIAN_STD_DEVIATION
     */
    pub fn from_random(rows: usize, cols: usize) -> Matrix {
        let mut data = vec![vec![0.0f32; cols]; rows];
        let rng = Normal::new(GAUSSIAN_MEAN, GAUSSIAN_STD_DEVIATION).unwrap();
        
        for i in 0..rows {
            for j in 0..cols {
                data[i][j] = rng.sample(&mut rand::thread_rng());
            }
        }

        Matrix {
            data,
            rows,
            cols
        }
    }


    /**
     * 1 Row, Many Columns
     */
    pub fn from_vec(cols: usize, data: Vec<f32>) -> Matrix {
        Matrix {
            data: vec![data],
            rows: 1,
            cols
        }
    }

    pub fn from_2d_vec(rows: usize, cols: usize, data: Vec<Vec<f32>>) -> Matrix {
        Matrix {
            data,
            rows,
            cols
        }
    }

    /**
     * Generate a new Matrix where all elements are x
     */
    pub fn fill_with(x: f32, rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: vec![vec![x; cols]; rows],
            rows,
            cols
        }
    }


    //-----------------------------------------------------
    // Non-overloaded Non-Static Linear Algebra Operations:
    //-----------------------------------------------------


    pub fn dot_product(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::fill_with(0.0f32, self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..other.rows {
                    result[i][j] += self[i][k] * other[k][j];
                }
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposition = Matrix::fill_with(0.0f32, self.cols, self.rows);

        for x in 0..self.rows {
            for y in 0..self.cols {
                transposition[y][x] = self[x][y];
            }
        }
        transposition
    }
    

    //-------------------------------------------------
    // Non-overloaded Static Linear Algebra Operations:
    //-------------------------------------------------


    /**
     * Multiply all of m by scalar:
     */
    pub fn apply_scalar(m: &Matrix, scalar: f32) -> Matrix {
        let mut result = m.clone();

        for x in 0..m.rows {
            for y in 0..m.cols {
                result[x][y] *= scalar;
            }
        }
        result
    }


    /**
     * Elementwise sum of entire Matrix
     */
    pub fn sum(m: &Matrix) -> f32 {
        let mut total = 0.0f32;
        
        for x in 0..m.rows {
            for y in 0..m.cols {
                total += m[x][y];
            }
        }
        total
    }


    pub fn max_of(val: f32, m: &Matrix) -> Matrix {
        let mut result = m.clone();

        for x in 0..m.rows {
            for y in 0..m.cols {
                if val > m[x][y] {
                    result[x][y] = val;
                }
            }
        }
        result 
    }

    pub fn colwise_sum_maintain_dim(m: &Matrix) -> Matrix {
        let mut result = m.clone();

        for x in 0..m.rows {
            let mut total = 0.0f32;
            for y in 0..m.cols {
                total += m[x][y];
            }
            
            result[x] = vec![total; m.cols];
        }
        result
    }


    pub fn one_hot_encode(rows: usize, cols: usize, hot: usize) -> Matrix {
        let mut classification = Matrix::fill_with(0.0f32, rows, cols);
        classification[hot][0] = 1.0f32;
        classification
    }

    pub fn one_hot_encode_by_maximum(m: &Matrix) -> Matrix {
        let mut classification = Matrix::fill_with(0.0f32, m.rows, m.cols);
        let mut maximum = m[0][0];
        let mut max_row_index = 0;

        for x in 1..m.rows {
            if m[x][0] > maximum {
                maximum = m[x][0];
                max_row_index = x;
            }
        }

        classification[max_row_index][0] = 1.0f32;
        classification
    }


    /**
     * e^x for all x in m
     */
    pub fn exp(m: &Matrix) -> Matrix {
        let mut result = m.clone();

        for x in 0..m.rows {
            for y in 0..m.cols {
                result[x][y] = f32::exp(m[x][y]);
            }
        }
        result
    }
}


//-------------------
// Utility Overloads:
//-------------------

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "[")?;
        for i in 0..(self.rows - 1) {
            write!(f, "{:?}\n ", self[i])?;
        }
        write!(f, "{:?}]", self[self.rows - 1])
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f32>;

    fn index(&self, x: usize) -> &Self::Output {
        &self.data[x]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, x: usize) -> &mut Self::Output {
        &mut self.data[x]
    }
}

impl PartialEq<Matrix> for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if (self.rows != other.rows) || (self.cols != other.cols) {
            return false;
        }

        for x in 0..self.rows {
            for y in 0..other.cols {
                if self[x][y] != other[x][y] {
                    return false;
                }
            }
        }

        true
    }
}


//----------------------
// Arithmetic Overloads:
//----------------------

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, m: &'b Matrix) -> Matrix {
        let mut result = self.clone();

        for x in 0..self.rows {
            for y in 0..m.cols {
                result[x][y] += m[x][y];
            }
        }
        result
    }
}


impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, m: &'b Matrix) -> Matrix {
        let mut result = self.clone();

        for x in 0..self.rows {
            for y in 0..m.cols {
                result[x][y] -= m[x][y];
            }
        }
        result
    }
}


impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, m: &'b Matrix) -> Matrix {
        let mut result = self.clone();

        for x in 0..self.rows {
            for y in 0..m.cols {
                result[x][y] *= m[x][y];
            }
        }
        result
    }
}


impl<'a, 'b> Div<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn div(self, m: &'b Matrix) -> Matrix {
        let mut result = self.clone();

        for x in 0..self.rows {
            for y in 0..m.cols {
                result[x][y] /= m[x][y];
            }
        }
        result
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, other: Self) {
        for x in 0..self.rows {
            for y in 0..self.cols {
                self[x][y] -= other[x][y];
            }
        }
    }
}