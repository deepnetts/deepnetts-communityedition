package deepnetts.tensor;


import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Static utility methods for tensors.
 *
 */
public class Tensors {
    // todo: add statistic functions: var, std, min, max, range, mean, q1, q3, etc.

//    public static Tensor of(float[] arr, int[] shape) {
//        Tensor t =new Tensor(arr);
//        new Tensor
//        return t;
//    }
    
    public static float[] copyOf(float[] arr) {
        float[] copy = new float[arr.length];
        System.arraycopy(arr, 0, copy, 0, arr.length);
        return copy;
    }

    // TODO: also set dimensions for dst
    public static final void copy(final Tensor src, final Tensor dest) {
        // todo: check dimensions and shape
        if (src.shape().equals(dest.shape())) {
            System.arraycopy(src.getValues(), 0, dest.getValues(), 0, src.getValues().length);
        } else {
            throw new DeepNettsException("Tensor shapes do not match!");
        }
    }

    public static final void copy(final float[] src, final float[] dest) {
        System.arraycopy(src, 0, dest, 0, src.length);
    }

    public static void sub(float[] arr, float val) {
        for(int i=0; i < arr.length; i++) {
            arr[i] = arr[i] - val;
        }
    }

    public static void multiply(float[] arr1, float[] arr2) {
        for (int i = 0; i < arr1.length; i++) {
            arr1[i] *= arr2[i];
        }
    }

    public static void fillFourthDim(final Tensor4D deltaWeights, final int fourthIdx, final float val) {
        for (int row = 0; row < deltaWeights.rows(); row++) {
            for (int col = 0; col < deltaWeights.cols(); col++) {
                for (int z = 0; z < deltaWeights.depth(); z++) {
                    deltaWeights.set(val, fourthIdx, z, row, col);
                }
            }
        }
    }

    /**
     * Prevent instantiation of this class.
     */
    private Tensors() { }

    /**
     * Returns tensors with max value for each component of input tensors.
     *
     * @param t
     * @param max proposed max tensor
     * @return tensor with max value for each component
     */
    public static TensorBase absMax(final Tensor t, final Tensor max) {
        final float[] tValues= t.getValues();
        final float[] maxValues= max.getValues();

        for(int i=0; i < tValues.length; i++) {
            if (Math.abs(tValues[i]) > maxValues[i]) maxValues[i] = Math.abs(tValues[i]);
            }
        return (TensorBase)max;       
        }

    /**
     * Returns array with max values for each position in the given input vectors.
     * Stores max values in second parameter.
     *
     * @param arr
     * @param max
     * @return
     */
    public static float[] absMax(final float[] arr, final float[] max) {
        for(int i=0; i < max.length; i++) {
            if (Math.abs(arr[i]) > max[i]) max[i] = Math.abs(arr[i]);
            }
        return max;
    }

    public static TensorBase absMin(final TensorBase t, final TensorBase min) {
        final float[] tValues= t.getValues();
        final float[] minValues= min.getValues();

        for(int i=0; i < tValues.length; i++) {
            if (Math.abs(tValues[i]) < Math.abs(minValues[i])) minValues[i] = Math.abs(tValues[i]);
            }
        return min;
    }

    public static float[] absMin(final float[] arr, final float[] min) {
        for(int i=0; i < arr.length; i++) {
            if (Math.abs(arr[i]) < Math.abs(min[i])) min[i] = Math.abs(arr[i]);
            }
        return min;
    }

    public static void div(final float[] array, final float val) {
        for (int i = 0; i < array.length; i++) {
            array[i] /= val;
        }
    }

    public static void div(final float[] array, final float[] divisor) {
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] / divisor[i];
        }
    }

    
    /**
     * Dot product matrix vector multiplication C = A . B
     *
     * @param resultC
     * @param matrixA
     * @param vectorB
     */
    public static Tensor1D dotProduct(Tensor2D matrixA, Tensor1D vectorB, Tensor1D resultC) {

        //if (matrixA.cols() != vectorB.numElements()) throw new RuntimeException("Number of columns in matrix A must be same as number of vector elements");
        
        float prodSum;
        for (int rowA = 0; rowA < matrixA.rows(); rowA++) {
            prodSum = 0;
            for (int colA = 0; colA < matrixA.cols(); colA++) {
                prodSum += matrixA.get(rowA, colA) * vectorB.get(colA);
            }
            resultC.set(prodSum, rowA);
        }

        return resultC;
    }

    // todo:  implement using strassen algorithm - matrica mora biti stepen broja 2
    /**
     * Matrix vector dot product.
     * Using matrix row buffering to accelerate computation.
     * @param matrixA
     * @param vectorB
     * @param resultC
     */
    public static void dotProductBuffered(Tensor2D matrixA, Tensor1D vectorB, Tensor1D resultC) {

        // if (matrixA.cols() != vectorB.numElements()) throw new DeepNettsException("Number of columns in matrix A must be same as number of vector elements");

        float prodSum;
        float[] rowBuff = new float[matrixA.cols()];
        float[] vectBuff = vectorB.getValues();
        float[] res = resultC.getValues();

        for (int rowA = 0; rowA < matrixA.rows(); rowA++) {
            rowBuff = matrixA.getRow(rowA, rowBuff);
            prodSum = 0;
            for (int colA = 0; colA < matrixA.cols(); colA++) {
                prodSum += rowBuff[colA] * vectBuff[colA];
            }
            res[rowA] = prodSum;
        }

    }

    
    // result = matrixA . matrixB
    // 
    public static void dotProduct(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result) {
        if (matrixA.cols() != matrixB.rows()) throw new DeepNettsException("Number of columns in matrix A does not match number of rows in matrix B! (They must be same)");

        for (int rowA = 0; rowA < matrixA.rows(); rowA++) {
            for (int colB = 0; colB < matrixB.cols(); colB++) {
                float prodSum = 0;
                for (int colA = 0; colA < matrixA.cols(); colA++) {
                    prodSum += matrixA.get(rowA, colA) * matrixB.get(colA, colB);
                }
                result.set(prodSum, rowA, colB);
            }
        }
    }

    public static void dotProduct(Tensor1D vectorA, Tensor1D vectorB, Tensor1D result) {
        if (vectorA.numElements()!= vectorB.numElements()) throw new DeepNettsException("Number of elements in vevtor A does not match number of elements in vector B! (They must be same)");

        float prodSum = 0;
        for (int idx = 0; idx < vectorA.numElements(); idx++) {
            prodSum += vectorA.get(idx) * vectorB.get(idx);
        }
        result.set(prodSum, 0);
    }

    public static void dotProductBuffered(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result) {
        //  if (matrixA.cols() != matrixB.rows()) throw new DeepNettsException("Number of columns in matrix A must be same as number of rows in matrix B");
        // svaka matrica treba da ima rows and cols cache - i to i h kesiraj tako da mogu da primenim starusen algoritamz amnozenje
        // to ce biti idealno i za vector api
        // pored toga kao i block matric u strausenovom i uradi i multitjreaded
        float prodSum;
        float[] rowBuffA = new float[matrixA.cols()];
        float[] colBuffB = new float[matrixB.rows()];

        for (int rowA = 0; rowA < matrixA.rows(); rowA++) {
            rowBuffA = matrixA.getRow(rowA, rowBuffA); // kreiraj row cache umesto ovog                            
            for (int colB = 0; colB < matrixB.cols(); colB++) {
                colBuffB = matrixB.getCol(colB, colBuffB); // napravi i col cache - kreiraj cache pre bilo kakve operacije                
                prodSum = 0;
                for (int colA = 0; colA < matrixA.cols(); colA++) {
                    prodSum += rowBuffA[colA] * colBuffB[colA];
                }
                result.set(prodSum, rowA, colB);
            }
        }
    }

    
    
    
    public static final void sub(final float[] array1, final float[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array1[i] -= array2[i];
        }
    }

    /**
     * Adds given vector and store result in first.
     *
     * @param array1
     * @param array2
     */
    public static final void add(final float[] array1, final float[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array1[i] += array2[i];
        }
    }

    // decimal scale
    
    // return min max mean std in one 
//    public static Tensor stats(Tensor t, Tensor min) {
//        final float[] tValues= t.getValues();
//        final float[] minValues= min.getValues();
//        
//        for(int i=0; i < tValues.length; i++) {
//            if (tValues[i] > minValues[i]) minValues[i] = tValues[i];
//        }
//        return min;       
//    }        
      
    public static Tensor1D zeros(int size) {
        final Tensor1D t = new Tensor1D(size);
        t.fill(0);
        return t;
    }

    public static Tensor1D ones(int size) {
        final Tensor1D t = new Tensor1D(size);
        t.fill(1f);
        return t;
    }

    /**
     * Generates a random 1D tensor with the specified dimensions.
     * Random values are uniformly distributed in range [0, 1]
     *
     * @param size size of the 1 dim tensor
     * @return 1D tensor filled with random values from global random num generator.
     */
    public static Tensor1D random(final int size) {
        Tensor1D tensor = new Tensor1D(size);

        for (int i = 0; i < tensor.numElements(); i++) {
            tensor.set(RandomGenerator.getDefault().nextFloat(), i);
        }

        return tensor;
    }

    /**
     * Create and return a tensor with specified number of rows and cols filled with random values.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @return  a random tensor of specificed dimensions
     */
    public static Tensor2D random(int rows, int cols) {
        Tensor2D tensor = new Tensor2D(rows, cols);

        for (int r = 0; r < tensor.rows(); r++) {
            for (int c = 0; c < tensor.cols(); c++) {
                tensor.set(RandomGenerator.getDefault().nextFloat(), r, c);
            }
        }
        return tensor;
    }

    public static Tensor3D random(int depth, int rows, int cols) {
        Tensor3D tensor = new Tensor3D(depth, rows, cols);

        for (int z = 0; z < tensor.depth(); z++) {
            for (int r = 0; r < tensor.rows(); r++) {
                for (int c = 0; c < tensor.cols(); c++) {
                    tensor.set(RandomGenerator.getDefault().nextFloat(), z, r, c);
                }
            }
        }
        return tensor;
    }

    public static Tensor4D random(int fourthDim, int depth, int rows, int cols) {
        Tensor4D tensor = new Tensor4D(fourthDim, depth, rows, cols );

        // ma samo izvrti sve indekse u values i postavi random vrednosti
        for (int f = 0; f < tensor.fourthDim(); f++) {
            for (int z = 0; z < tensor.depth(); z++) {
                for (int r = 0; r < tensor.rows(); r++) {
                    for (int c = 0; c < tensor.cols(); c++) {
                        tensor.set(RandomGenerator.getDefault().nextFloat(), f, z, r, c);
                    }
                }
            }
        }
        return tensor;
    }

    public static void min(Tensor t1, Tensor t2, Tensor result) {
        int size = t1.numElements();
        final float[] arr1 = t1.getValues();
        final float[] arr2 = t2.getValues();
        final float[] res = result.getValues();

        for(int i=0; i<size; i++) {
            res[i] = Math.min(arr1[i], arr2[i]);
        }

    }

    public static void max(Tensor t1, Tensor t2, Tensor result) {
        int size = t1.numElements();

        final float[] arr1 = t1.getValues();
        final float[] arr2 = t2.getValues();
        final float[] res = result.getValues();

        for(int i=0; i<size; i++) {
            res[i] = Math.max(arr1[i], arr2[i]);
        }
    }

    /**
     * Factory method for creating tensor instance,
     *
     * @param rows
     * @param cols
     * @param values
     * @return
     */
    public static Tensor2D create(int rows, int cols, float[] values) {
        return new Tensor2D(rows, cols, values);
    }

    public static Tensor3D create(int depth, int rows, int cols, float[] values) {
        return new Tensor3D(depth, rows, cols,  values);
    }

    public static Tensor4D create(int fourthDim, int depth, int rows, int cols, float[] values) {
        return new Tensor4D(fourthDim, depth, rows, cols, values);
    }

    public static Tensor fromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);

        String dimLine = lines.get(0);
        String dimStr = dimLine.substring(dimLine.lastIndexOf(":")+1);
        int dim = Integer.parseInt(dimStr);
        String shapeStr = lines.get(1);
        shapeStr = shapeStr.substring(shapeStr.lastIndexOf(":")+1);
        int[] shape = parseInts(shapeStr);

        // values: in line 2
        
        // u zavisnosti od broja dimenzija ucitavaj i vracaj 1d, 2d, 3d, 4d tenzore
        //todoL load shape and values
        if (dim == 1) {
            String valuesStr = lines.get(3);
            float[] values = parseFloats(valuesStr);
            Tensor1D tensor = new Tensor1D(values);
            return tensor;
        } else if (dim == 2) {
            final int rows = shape[0];
            final int cols = shape[1];
            Tensor2D tensor = new Tensor2D(rows, cols);
            int rowIdx = 0;
            for(int i=3; i< lines.size(); i++) {
                float[] rowVals = parseFloats(lines.get(i));
                tensor.setRow(rowIdx, rowVals);
                rowIdx++;
            }
            return tensor;
        } else if (dim == 3) {
            int depth = shape[0];
            int rows = shape[1];
            int cols = shape[2];
            Tensor3D tensor = new Tensor3D(depth, rows, cols);
            int rowIdx = 0, depthIdx=0;
            for(int i=3; i< lines.size(); i++) {
                float[] rowVals = parseFloats(lines.get(i));
                tensor.setRow(rowVals, depthIdx, rowIdx);
                rowIdx++;
                if (rowIdx == rows) {
                    rowIdx=0;
                    depthIdx++;
                }
            }
            return tensor;

        } else if (dim == 4) {
            int fourthDim = shape[0];
            int depth = shape[1];
            int rows = shape[2];
            int cols = shape[3];
            Tensor4D tensor = new Tensor4D(fourthDim, depth, rows, cols);
            int rowIdx = 0, depthIdx=0, fourthIdx=0;
            for(int i=3; i< lines.size(); i++) {
                float[] rowVals = parseFloats(lines.get(i));
                tensor.setRow(rowVals, fourthIdx, depthIdx, rowIdx);
                rowIdx++;
                if (rowIdx == rows) {
                    rowIdx=0;
                    depthIdx++;
                }

                if (depthIdx == depth) {
                    depthIdx = 0;
                    fourthIdx++;
                }
            }
            return tensor;
        }

        throw new DeepNettsException("Error reading tensor from file: "+filePath);

    }

    public static int[] parseInts(String str) {
        String[] parts = str.split(",");
        int[] values = new int[parts.length];
        for(int i =0; i<parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }

    public static float[] parseFloats(String str) {
        String[] parts = str.split(",");
        float[] values = new float[parts.length];
        for(int i =0; i<parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    }
