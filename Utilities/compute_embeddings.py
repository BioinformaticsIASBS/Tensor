import csv
import numpy as np


def save_to_file(dataset: str) -> None:
    if dataset == "DTINET":
        """
        Load protein interaction and similaritiy matrices
        """
        file = open("Data/DTINet dataset/mat_protein_protein.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        protein_interactions = np.array([list(map(int, row[1:])) for row in tail])

        file = open("Data/DTINet dataset/Similarity_Matrix_Proteins.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        protein_sequence_sim = np.array([list(map(float, row[1:])) for row in tail]) / 100

        file = open("Data/DTINet dataset/protein_similarity_disease.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        protein_disease_sim = np.array([list(map(float, row[1:])) for row in tail])

        """
        Load drug interaction and similaritiy matrices
        """
        file = open("Data/DTINet dataset/Similarity_Matrix_Drugs.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_structure_sim = np.array([list(map(float, row[1:])) for row in tail])

        file = open("Data/DTINet dataset/mat_drug_drug.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_ineractions = np.array([list(map(int, row[1:])) for row in tail])

        file = open("Data/DTINet dataset/drug_similarity_disease.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_disease_sim = np.array([list(map(float, row[1:])) for row in tail])

        file = open("Data/DTINet dataset/drug_similarity_se.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drug_se_sim = np.array([list(map(float, row[1:])) for row in tail])

        """
        Create direct embeddings
        """
        D = drug_structure_sim + drug_ineractions + drug_disease_sim + drug_se_sim
        T = protein_interactions + protein_sequence_sim + protein_disease_sim

        np.savetxt("Data/DTINet dataset/D.txt", D)
        np.savetxt("Data/DTINet dataset/T.txt", T)

    else:
        """
        Load drug similarity matrix
        """
        file = open("Data/Gold standard dataset/" + dataset.lower() + "_simmat_dc.txt")
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        D = np.array([list(map(float, row)) for row in tail])

        """
        Load target similarity matrix
        """
        file = open("Data/Gold standard dataset/" + dataset.lower() + "_simmat_dg.txt")
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        T = np.array([list(map(float, row)) for row in tail])

        np.savetxt("Data/Gold standard dataset/D.txt", D)
        np.savetxt("Data/Gold standard dataset/T.txt", T)
