import csv
import numpy as np
from scipy import spatial

def find_similarity_matrix(input_matrix: np.ndarray) -> np.ndarray:
    input_matrix_len = len(input_matrix)
    similarity_matrix = np.zeros((input_matrix_len, input_matrix_len))

    for i in range(input_matrix_len):
      for j in range(input_matrix_len):
        i_j_rows_similarity = 1 - spatial.distance.cosine(input_matrix[i], input_matrix[j])
        similarity_matrix[i][j] = i_j_rows_similarity

    return similarity_matrix

def save_to_file() -> None:
    '''
    Compute drug similarity based on the drug-disease matrix
    '''
    in_file = open('Data/DTINet dataset/mat_drug_disease.csv')
    csvreader = csv.reader(in_file)
    data = [row for row in csvreader]
    head = data[0]
    tail = data[1:]
    diseases = head[1:]
    drugs = [row[0] for row in tail]
    drug_disease = np.array([list(map(int, row[1:])) for row in tail])
    similarity_matrix = find_similarity_matrix(drug_disease)

    out_file = open('Data/DTINet dataset/drug_similarity_disease.csv', 'w', newline='')
    csvwriter = csv.writer(out_file)
    csvwriter.writerow([''] + drugs)
    i = 0
    for row in similarity_matrix:
        csvwriter.writerow([drugs[i]] + row.tolist())
        i += 1
    out_file.close()


    '''
    Compute drug similarity based on the drug-side effect matrix
    '''
    in_file = file = open('Data/DTINet dataset/mat_drug_se.csv')
    csvreader = csv.reader(in_file)
    data = [row for row in csvreader]
    head = data[0]
    tail = data[1:]
    side_effects = head[1:]
    drugs = [row[0] for row in tail]
    drug_se = np.array([list(map(int, row[1:])) for row in tail])
    similarity_matrix = find_similarity_matrix(drug_se)

    out_file = open('Data/DTINet dataset/drug_similarity_se.csv', 'w', newline='')
    csvwriter = csv.writer(out_file)
    csvwriter.writerow([''] + drugs)
    i = 0
    for row in similarity_matrix:
        csvwriter.writerow([drugs[i]] + row.tolist())
        i += 1
    out_file.close()


    '''
    Compute protein similarity based on the protein-disease matrix
    '''
    in_file = open('Data/DTINet dataset/mat_protein_disease.csv')
    csvreader = csv.reader(in_file)
    data = [row for row in csvreader]
    head = data[0]
    tail = data[1:]
    diseases = head[1:]
    proteins = [row[0] for row in tail]
    protein_disease = np.array([list(map(int, row[1:])) for row in tail])
    similarity_matrix = find_similarity_matrix(protein_disease)

    out_file = open('Data/DTINet dataset/protein_similarity_disease.csv', 'w', newline='')
    csvwriter = csv.writer(out_file)
    csvwriter.writerow([''] + proteins)
    i = 0
    for row in similarity_matrix:
        csvwriter.writerow([proteins[i]] + row.tolist())
        i += 1
    out_file.close()

