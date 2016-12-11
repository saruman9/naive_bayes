#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rulinalg;

mod gaussian;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::utils;

use std::io;

fn main() {
    env_logger::init().unwrap();

    // Read count of train elements and count of features.
    println!("Input count of train elements and count of features (e.g. 20 2):");
    let mut properties = String::new();
    io::stdin().read_line(&mut properties).expect("Error of reading properties of train data.");
    let mut properties = properties.split_whitespace();
    let count_elements: usize = properties.next()
        .and_then(|x| x.parse().ok())
        .expect("Error of read count of elements of train data.");
    let count_features: usize = properties.next()
        .and_then(|x| x.parse().ok())
        .expect("Error of read count of features of train data.");
    println!("Count of elements: {}; count of features: {}.", count_elements, count_features);
    // Read train data.
    println!("Input train data (e.g.\n1.332 234.2\n2.34 3\n4 5):");
    let mut train_data: Vec<f64> = Vec::new();
    for _ in 0..count_elements {
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).expect("Error of reading train data.");
        let mut buf = buf.split_whitespace();
        while let Some(buf) = buf.next() {
            train_data.push(buf.parse().expect("Error of parsing train data."));
        }
    }

    let train = Matrix::new(count_elements, count_features, train_data);
    println!("Train data:\n{}", train);

    // Read count of classes.
    println!("Input count of classes:");
    let mut count_classes = String::new();
    io::stdin().read_line(&mut count_classes).expect("Error of reading count of classes.");
    let count_classes: usize = count_classes.trim().parse()
        .expect("Error of parsing count of classes.");
    // Read data of class.
    let mut class_data: Vec<f64> = Vec::new();
    println!("Input classes data (e.g.\n1 0 0\n0 0 1\n0 1 0):");
    for _ in 0..count_elements {
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).expect("Error of reading class data.");
        let mut buf = buf.split_whitespace();
        while let Some(buf) = buf.next() {
            class_data.push(buf.parse().expect("Error of parsing class data."));
        }
    }

    let class = Matrix::new(count_elements, count_classes, class_data);
    println!("Class data:\n{}", class);

    // Read count of targets.
    println!("Input count of targets for predict:");
    let mut count_targets = String::new();
    io::stdin().read_line(&mut count_targets).expect("Error of reading count of target data.");
    let count_elements_target: usize = count_targets.trim().parse()
        .expect("Error of parsing count of elements of target data.");
    // Read data of targets.
    println!("Count of elements: {}; count of features: {}", count_elements_target, count_features);
    println!("Input target data (e.g.\n1.332 234.2 3.4\n2.34 3 5.6\n4 5 0.0):");
    let mut target_data: Vec<f64> = Vec::new();
    for _ in 0..count_elements_target {
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).expect("Error of reading target data.");
        let mut buf = buf.split_whitespace();
        while let Some(buf) = buf.next() {
            target_data.push(buf.parse().expect("Error of parsing target data."));
        }
    }

    let targets = Matrix::new(count_elements_target, count_features, target_data);
    println!("Targets data:\n{}", targets);

    let class_count = class.cols();
    let feature_count = train.cols();
    let data_counts = train.rows() as f64;

    let mut class_counts = vec![0; class_count];
    let mut class_data: Vec<Vec<usize>> = vec![Vec::new(); class_count];

    let mut g = gaussian::Gaussian::from_model(class_count, feature_count);

    // Detect rows of the train data, that match specific class.
    for (num_row, row) in class.iter_rows().enumerate() {
        let mut class: i32 = -1;
        for (num_col, c) in row.into_iter().enumerate() {
            if *c == 1f64 {
                class = num_col as i32;
            }
        }

        // If class not detect then will be panic because index is -1.
        class_data[class as usize].push(num_row);
        class_counts[class as usize] += 1;
    }
    debug!("class_data: {:?}", class_data);
    debug!("class_counts: {:?}", class_counts);

    // Compute Gaussian function for every class and feature.
    for (num_class, rows_of_inputs) in class_data.into_iter().enumerate() {
        g.compute_gaussian(&train.select_rows(&rows_of_inputs), num_class);
    }

    debug!("Computed Gaussian: {:?}", g);

    // Compute Class Prior Probability (P(c)) for classes.
    let class_prior: Vec<f64> = class_counts.iter().map(|c| *c as f64 / data_counts).collect();
    debug!("Class Prior Probability: {:?}", class_prior);

    let result = g.compute_likehood_and_predict(&targets, &class_prior);
    debug!("Result:\n{}", result);
    for (target_num, row) in result.iter_rows().enumerate() {
        println!("Class of target #{} is {}",
                 target_num,
                 utils::argmax(row).0 + 1);
    }
}

#[cfg(test)]
mod tests {
    use gaussian;
    use rulinalg::matrix::{Matrix, BaseMatrix};
    use rulinalg::utils;

    #[test]
    fn test_gaussian() {
        let train = Matrix::new(6,
                                2,
                                vec![1.0, 1.1, 1.1, 0.9, 2.2, 2.3, 2.5, 2.7, 5.2, 4.3, 6.2, 7.3]);

        let class = Matrix::new(6,
                                3,
                                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let targets = Matrix::new(6,
                                  2,
                                  vec![1.0, 1.1, 1.1, 0.9, 2.2, 2.3, 2.5, 2.7, 5.2, 4.3, 6.2, 7.3]);

        let class_count = class.cols();
        let feature_count = train.cols();
        let data_counts = train.rows() as f64;

        let mut class_counts = vec![0; class_count];
        let mut class_data: Vec<Vec<usize>> = vec![Vec::new(); class_count];

        let mut g = gaussian::Gaussian::from_model(class_count, feature_count);

        // Detect rows of the train data, that match specific class.
        for (num_row, row) in class.iter_rows().enumerate() {
            let mut class: i32 = -1;
            for (num_col, c) in row.into_iter().enumerate() {
                if *c == 1f64 {
                    class = num_col as i32;
                }
            }

            // If class not detect then will be panic because index is -1.
            class_data[class as usize].push(num_row);
            class_counts[class as usize] += 1;
        }
        debug!("class_data: {:?}", class_data);
        debug!("class_counts: {:?}", class_counts);

        // Compute Gaussian function for every class and feature.
        for (num_class, rows_of_inputs) in class_data.into_iter().enumerate() {
            g.compute_gaussian(&train.select_rows(&rows_of_inputs), num_class);
        }

        debug!("Computed Gaussian: {:?}", g);

        // Compute Class Prior Probability (P(c)) for classes.
        let class_prior: Vec<f64> = class_counts.iter().map(|c| *c as f64 / data_counts).collect();
        debug!("Class Prior Probability: {:?}", class_prior);

        let result = g.compute_likehood_and_predict(&targets, &class_prior);
        debug!("Result:\n{}", result);
        let outputs: Vec<usize> = result.iter_rows().map(|x| utils::argmax(x).0).collect();

        assert_eq!(outputs, [0, 0, 1, 1, 2, 2]);
    }
}
