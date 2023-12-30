import argparse
import csv
import numpy as np
from sklearn.model_selection import KFold
from Utilities import compute_similarity_matrices, dtinet_optimizer, gs_optimizer


def TMTF(args: argparse.Namespace) -> np.ndarray:
    if args.dataset == 'DTINET':
        if args.comp_sim_mats:
            compute_similarity_matrices.save_to_file()
            
        '''
        Load drug-target matrix
        '''
        file = open('Data/DTINet dataset/mat_drug_protein.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        proteins = head[1:]
        drugs = [row[0] for row in tail]
        X = np.array([list(map(int, row[1:])) for row in tail])

        '''
        Load target protein similaritiy matrices
        '''
        file = open('Data/DTINet dataset/mat_protein_protein.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        protein_interactions = np.array([list(map(int, row[1:])) for row in tail])

        file = open('Data/DTINet dataset/Similarity_Matrix_Proteins.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        protein_sequence_sim = np.array([list(map(float, row[1:])) for row in tail])/100

        file = open('Data/DTINet dataset/protein_similarity_disease.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        protein_disease_sim = np.array([list(map(float, row[1:])) for row in tail])

        '''
        Load drug similaritiy matrices
        '''
        file = open('Data/DTINet dataset/Similarity_Matrix_Drugs.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_structure_sim = np.array([list(map(float, row[1:])) for row in tail])

        file = open('Data/DTINet dataset/mat_drug_drug.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_ineractions = np.array([list(map(int, row[1:])) for row in tail])

        file = open('Data/DTINet dataset/drug_similarity_disease.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_disease_sim = np.array([list(map(float, row[1:])) for row in tail])

        file = open('Data/DTINet dataset/drug_similarity_se.csv')
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_se_sim = np.array([list(map(float, row[1:])) for row in tail])

        '''
        Create tensors
        '''
        script_T = np.concatenate((np.expand_dims(protein_sequence_sim, axis=-1),
                                      np.expand_dims(protein_interactions, axis=-1),
                                      np.expand_dims(protein_disease_sim, axis=-1)), axis=-1)

        script_D = np.concatenate((np.expand_dims(drug_structure_sim, axis=-1),
                                      np.expand_dims(drug_ineractions, axis=-1),
                                      np.expand_dims(drug_disease_sim, axis=-1),
                                      np.expand_dims(drug_se_sim, axis=-1)), axis=-1)

        '''
        Cross-validate and predict
        '''
        pos_sample_indices = np.where(X == 1)
        neg_sample_indices = np.where(X == 0)
        pos_sample_splits = KFold(n_splits=args.k, shuffle=True).split(pos_sample_indices[0])
        neg_sample_splits = KFold(n_splits=args.k, shuffle=True).split(neg_sample_indices[0])
        X_hat = np.zeros(X.shape)

        fold_no = 1
        log_file = open('log.txt', 'w') if args.log else None
        for _, pos_test_split in pos_sample_splits:
            log_file.write('Fold {}\n'.format(fold_no)) if args.log else None
            _, neg_test_split = next(neg_sample_splits)

            pos_test_indices = (pos_sample_indices[0][pos_test_split], pos_sample_indices[1][pos_test_split])
            test_indices = (np.concatenate((neg_sample_indices[0][neg_test_split],
                                            pos_sample_indices[0][pos_test_split]), axis=0),
                             np.concatenate((neg_sample_indices[1][neg_test_split],
                                             pos_sample_indices[1][pos_test_split]), axis=0))
            X_k = np.copy(X)
            X_k[pos_test_indices] = 0

            X_k_hat = dtinet_optimizer.optimize(X_k, script_D, script_T, args.form_no, args.f_size,
                                                args.alpha, args.epsilon, test_indices, log_file)

            X_hat[test_indices] = X_k_hat[test_indices]
            fold_no += 1

        np.savetxt('X_hat.txt', X_hat)
        log_file.close() if args.log else None


    
    else:
        '''
        Load drug-target matrix
        '''
        file = open('Data/Gold standard dataset/' + args.dataset.lower() + '_admat_dgc.txt')
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        X = np.array([list(map(int, row)) for row in tail])

        '''
        Load drug similarity matrix
        '''
        file = open('Data/Gold standard dataset/' + args.dataset.lower() + '_simmat_dc.txt')
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        drug_str_sim = np.array([list(map(float, row)) for row in tail])

        '''
        Load target similarity matrix
        '''
        file = open('Data/Gold standard dataset/' + args.dataset.lower() + '_simmat_dg.txt')
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        protein_seq_sim = np.array([list(map(float, row)) for row in tail])

        '''
        Cross-validate and predict
        '''
        pos_sample_indices = np.where(X == 1)
        neg_sample_indices = np.where(X == 0)
        pos_sample_splits = KFold(n_splits=args.k, shuffle=True).split(pos_sample_indices[0])
        neg_sample_splits = KFold(n_splits=args.k, shuffle=True).split(neg_sample_indices[0])
        X_hat = np.zeros(X.shape)

        fold_no = 1
        log_file = open('log.txt', 'w') if args.log else None
        for _, pos_test_split in pos_sample_splits:
            log_file.write('Fold {}\n'.format(fold_no)) if args.log else None
            _, neg_test_split = next(neg_sample_splits)

            pos_test_indices = (pos_sample_indices[0][pos_test_split], pos_sample_indices[1][pos_test_split])
            test_indices = (np.concatenate((neg_sample_indices[0][neg_test_split],
                                            pos_sample_indices[0][pos_test_split]), axis=0),
                             np.concatenate((neg_sample_indices[1][neg_test_split],
                                             pos_sample_indices[1][pos_test_split]), axis=0))
            X_k = np.copy(X)
            X_k[pos_test_indices] = 0

            X_k_hat = gs_optimizer.optimize(X_k, drug_str_sim, protein_seq_sim, args.f_size,
                                            args.alpha, args.epsilon, test_indices, log_file)

            X_hat[test_indices] = X_k_hat[test_indices]
            fold_no += 1

        np.savetxt('X_hat.txt', X_hat)
        log_file.close() if args.log else None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['DTINET', 'E', 'GPCR', 'IC', 'NR'], help='Data to train/test the model on')
    parser.add_argument('--k', type=int, default=10, help='Number of folds in the k-fold cross-validation process')
    parser.add_argument('--form_no', type=int, default=1, choices=range(1,5), help='Formulation of the model')
    parser.add_argument('--f_size', type=int, default=16, help='Number of latent variables')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Stopping criterion')
    parser.add_argument('--comp_sim_mats', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to compute similarity matrices or not (Set True at least once)')
    parser.add_argument('--log', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to log the optimization process')

    args = parser.parse_args()
    X_hat = TMTF(args)
