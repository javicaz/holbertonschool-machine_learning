#!/usr/bin/env python3


def cat_matrices(mat1, mat2, axis=0):

    # Check if the dimensions of the matrices are compatible for concatenation
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Concatenate along rows (axis 0)
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        # Concatenate along columns (axis 1)
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        # Invalid axis value
        return None


# Test the function with sample matrices
matrix1 = [[1, 2, 3], [4, 5, 6]]
matrix2 = [[7, 8, 9], [10, 11, 12]]

result = cat_matrices(matrix1, matrix2, axis=0)
print("Concatenated along axis 0:")
print(result)

result = cat_matrices(matrix1, matrix2, axis=1)
print("Concatenated along axis 1:")
print(result)
