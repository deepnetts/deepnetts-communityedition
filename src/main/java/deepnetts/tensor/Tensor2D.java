package deepnetts.tensor;

import deepnetts.core.DeepNetts;
import static deepnetts.tensor.Tensors.parseFloats;
import static deepnetts.tensor.Tensors.parseInts;
import deepnetts.util.DeepNettsException;
import deepnetts.util.DeepNettsThreadPool;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;


/**
 * A 2D tensor / matrix with specified number of rows and columns..
 */
public final class Tensor2D extends TensorBase {

    private final int rows, cols;
    private float[][] rowsCache, colsCache;
    
    private DeepNettsThreadPool threadPool;

    // private final Layout layout = Layout.NCHW;
    /**
     * Creates a tensor with specified number of rows and columns (matrix).
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Tensor2D(int rows, int cols) {
        super(Shape.of(rows, cols));
        if (rows <= 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative or zero: " + rows);
        }
        if (cols <= 0) {
            throw new IllegalArgumentException("Number of columns cannot be negative or zero: " + cols);
        }

        this.rows = rows;
        this.cols = cols;
        values = new float[rows * cols];
    }

    public Tensor2D(int rows, int cols, float[] values) {
        super(Shape.of(rows, cols));
        if (rows <= 0) {
            throw new IllegalArgumentException("Number of rows cannot be zero or negative: " + rows);
        }
        if (cols <= 0) {
            throw new IllegalArgumentException("Number of cols cannot be zero or negative: " + cols);
        }
        if (rows * cols != values.length) {
            throw new IllegalArgumentException("Number of values does not match tensor dimensions! " + values.length);
        }

        this.rows = rows;
        this.cols = cols;
        this.values = values; // @layoutfix: ovde bi values morao da bude u column first formatu - to zavisi od toga sta mu se prosledi
    }

    /**
     * Creates a 2D tensor / matrix filled with given 2d array. First dimension
     * corresponds to number of rows, while second dimension corresponds to
     * number of columns.
     *
     * @param vals values to store into matrix
     */
    public Tensor2D(final float[][] vals) {
        super(Shape.of(vals.length, vals[0].length));
        this.rows = vals.length;
        this.cols = vals[0].length;
        this.values = new float[rows * cols];

        // copy array values into tensor
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                set(vals[row][col], row, col);
            }
        }
    }

    /**
     * Returns a number of columns in this matrix.
     *
     * @return
     */
    public final int cols() {
        return cols;
    }

    /**
     * Returns a number of rows in this matrix.
     *
     * @return
     */
    public final int rows() {
        return rows;
    }

    /**
     * Returns a value at specified row, col position in this tensor.
     *
     * @param row
     * @param col
     * @return value at [row, col]
     */
    public final float get(final int row, final int col) {
        final int idx = col * rows + row;
        return values[idx];
    }

    /**
     * Set value at specified [row, col] position in this tensor.
     *
     * @param val value to set
     * @param row tensor's row index
     * @param col tensor's col index
     */
    public final void set(final float val, final int row, final int col) {
        final int idx = col * rows + row;
        values[idx] = val;
    }

    public void setRow(final int rowIdx, final float[] rowData) {
        if (rowData.length != cols) {
            throw new RuntimeException("Row length does not match the number of tensor columns:" + rowData.length);
        }
        if (rowIdx >= rows) {
            throw new RuntimeException("Row index is greater then the  number of tensor rows:" + rowIdx);
        }

        for (int colIdx = 0; colIdx < rowData.length; colIdx++) {
            set(rowData[colIdx], rowIdx, colIdx);
        }
    }

    public void setCol(final int colIdx, final float[] colData) {
        if (colData.length != rows) {
            throw new RuntimeException("Col length does not match the number of tensor rows:" + colData.length);
        }
        if (colIdx >= cols) {
            throw new RuntimeException("Col index is greater then the  number of tensor rows:" + colIdx);
        }

        for (int rowIdx = 0; rowIdx < colData.length; rowIdx++) {
            set(colData[rowIdx], rowIdx, colIdx);
        }

    }

    public void createRowsCache() {
        rowsCache = new float[rows][cols];
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            float[] row = new float[cols];
            rowsCache[rowIdx] = getRow(rowIdx, row);
        }
    }

    public void createColsCache() {
        colsCache = new float[cols][rows];
        for (int c = 0; c < cols; c++) {
            float[] col = new float[rows];
            colsCache[c] = getCol(c, col);
        }
    }

    public void createRowsAndColsCache() {
        createRowsCache();
        createColsCache();
    }
    
    public float[] getRow(final int rowIdx, final float[] rowBuff) {
        // vrati vektor sa svim vrednostima u redu row, po svim kolonama c
        for (int idx = rowIdx, c = 0; c < cols; c++, idx += rows) { // idx < (row+1)*cols
            rowBuff[c] = values[idx];
        }

        return rowBuff;
    }

    public float[] getCol(final int col, final float[] colBuff) {

        /*for(int idx=col*rows, r=0; r<rows; r++, idx++) {
            colBuff[r] = values[idx];
        }*/
        System.arraycopy(this.values, col * rows, colBuff, 0, rows);

        return colBuff;
    }

    /**
     * Adds specified value to matrix value at position x, y
     *
     * @param col
     * @param row
     * @param value
     */
    public final void add(final float value, final int row, final int col) {
        final int idx = col * rows + row;
        values[idx] += value;
    }

    // todo : createTest
    public final Tensor2D add(final Tensor1D toAdd) { // by row ot by col, param?

        if (this.rows != toAdd.numElements() && this.cols != toAdd.numElements()) {
            throw new DeepNettsException("Lenghts don't match => 2D shape:" + this.shape + "Vector lenght: " + toAdd.numElements());
        }

        if (DeepNetts.getInstance().useVectorAPI()) {
            vectorizationImpl.addVectorized(this, toAdd);
            return this;
        }

        // add to cols
        if (this.rows == toAdd.numElements()) {
            for (int c = 0; c < cols; c++) {
                final int colOffset = c * rows;
                for (int i = 0; i < toAdd.numElements(); i++) {
                    this.values[colOffset + i] += toAdd.values[i];
                }
            }
        } else if (this.cols == toAdd.numElements()) { // add to rows
            for (int r = 0; r < rows; r++) {
                for (int i = 0; i < toAdd.numElements(); i++) {
                    this.values[r + i * rows] += toAdd.values[i]; //??? check ali mislim da je dobro
                }
            }
        } else {
            throw new DeepNettsException("Bad tensor size :" + toAdd.numElements());
        }

        return this;
    }
    
//    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
//    static final int vecLen = SPECIES.length();       

//    final Tensor2D addVectorized(final Tensor1D toAdd) {
//        if (this.rows == toAdd.numElements()) {
//            // Dodajemo vektor toAdd (dimenzije `rows`) na SVAKU kolonu (tj. instancu u batch-u)
//            for (int c = 0; c < cols; c++) {
//                int colOffset = c * rows;
//                int i = 0;
//                int upperBound = SPECIES.loopBound(rows);
//                for (; i < upperBound; i += vecLen) {
//                    FloatVector matVec = FloatVector.fromArray(SPECIES, this.values, colOffset + i);
//                    FloatVector addVec = FloatVector.fromArray(SPECIES, toAdd.values, i);
//                    FloatVector resVec = matVec.add(addVec);
//                    resVec.intoArray(this.values, colOffset + i);
//                }
//                // Skalarni ostatak
//                for (; i < rows; i++) {
//                    this.values[colOffset + i] += toAdd.values[i];
//                }
//            }
//
//        } else if (this.cols == toAdd.numElements()) {
//            for (int c = 0; c < cols; c++) {
//                float toAddVal = toAdd.values[c];
//                FloatVector toAddVec = FloatVector.broadcast(SPECIES, toAddVal);
//
//                int colOffset = c * rows;
//                int i = 0;
//                int upperBound = SPECIES.loopBound(rows);
//
//                for (; i < upperBound; i += vecLen) {
//                    FloatVector colVec = FloatVector.fromArray(SPECIES, this.values, colOffset + i);
//                    FloatVector resVec = colVec.add(toAddVec);
//                    resVec.intoArray(this.values, colOffset + i);
//                }
//
//                for (; i < rows; i++) {
//                    this.values[colOffset + i] += toAddVal;
//                }
//            }
//        } else {
//            throw new DeepNettsException("Invalid toAdd length: expected " + rows + " or " + cols + ", got " + toAdd.numElements());
//        }
//
//        return this;
//    }

    public final void sub(final float val, final int row, final int col) {
        final int idx = col * rows + row;
        values[idx] -= val;
    }

    @Override
    public TensorBase copy() {
        Tensor2D newTensor = new Tensor2D(rows, cols);
        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);
        return newTensor;
    }

    @Override
    public Object clone() {
        Tensor2D newTensor = new Tensor2D(rows, cols);
        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);
        return newTensor;
    }

    // todo: create method transpose into
    public Tensor2D getTransposed() {
        Tensor2D transposed = new Tensor2D(cols, rows);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float v = get(r, c);
                transposed.set(v, c, r);
            }
        }
        return transposed;
    }
    
    public Tensor2D transposeInto(Tensor2D transposed) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                final float v = get(r, c);
                transposed.set(v, c, r);
            }
        }
        return transposed;
    }
    

    @Override
    public void setValuesFromStringTransposed(String values) { // , int layoutOrd -> col, row ; nhw, nchw, nhwc
        // koji je layout u stringu a koji treba da bude u tenzoru
        // u stringu je layout row major
        // u tenzoru treba da je col major
        // po defaultu neka bude layoutOrd=0 row major, a layoutOrd=1 col major
        // problem je sto nije isti layout u tf exportu i u dn, mozda najbolje da resim export da bude kako treba
        int startPos = 0, nextPos = 0;
        int idx = 0, r = 0, c = 0;
        // rows and cols are available from attributes of this tensor
        // r and c are rows and columns from string
        while (idx < rows * cols) {
            nextPos = values.indexOf(",", startPos);
            if (nextPos == -1) {
                continue; // potencijalni bug za psolednju vrednost, radi dobro jer ima zarez pre biases
            }
            String strVal = values.substring(startPos, nextPos);
            float val = Float.parseFloat(strVal.trim());

            if (c == rows) { // jer je transponovano
                c = 0;
                r++;
            }

            this.set(val, c, r);

            startPos = nextPos + 1;
            idx++;
            // rows and cols counter
            c++;

        }
    }

//    @Override
//    public String toString() {
//        StringBuilder sb = new StringBuilder();
//
//        sb.append(this.shape.toString());
//        sb.append(" Values: [");
//        for (int r = 0; r < rows; r++) {
//            for (int c = 0; c < cols; c++) {
//                sb.append(get(r, c));
//                if (c < cols - 1) {
//                    sb.append(",");
//                }
//            }
//            sb.append("\n");
//        }
//        sb.append("]");
//
//        return sb.toString();
//    }
    public static Tensor2D fromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);

        String dimLine = lines.get(0);
        String dimStr = dimLine.substring(dimLine.lastIndexOf(":") + 1);
        int dim = Integer.parseInt(dimStr);
        String shapeStr = lines.get(1);
        shapeStr = shapeStr.substring(shapeStr.lastIndexOf(":") + 1);
        int[] shape = parseInts(shapeStr);

        if (dim == 2) {
            final int rows = shape[0];
            final int cols = shape[1];
            Tensor2D tensor = new Tensor2D(rows, cols);
            int rowIdx = 0;
            for (int i = 3; i < lines.size(); i++) {
                float[] row = parseFloats(lines.get(i));
                tensor.setRow(rowIdx, row);
                rowIdx++;
            }
            return tensor;
        }

        throw new DeepNettsException("Error reading 2D tensor from file: " + filePath);
    }

    public void toFile(String filePath) {

        StringBuilder sb = new StringBuilder();

        sb.append("dims:2").append("\n");
        sb.append("shape:").append(rows).append(",").append(cols).append("\n");
        sb.append("values:\n");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sb.append(get(i, j));
                if (j < cols - 1) {
                    sb.append(",");
                }
            }
            sb.append("\n");
        }

        Path path = Paths.get(filePath);
        try {
            Files.write(path, sb.toString().getBytes());
        } catch (IOException ex) {
            Logger.getLogger(Tensor2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public Tensor1D sqrSumByCol(Tensor1D sumSqr) {

        for (int c = 0; c < cols; c++) {
            float sum = 0f;
            for(int r=0; r< rows; r++) {
                sum += get(r, c);
            }
            sumSqr.set(sum, c);
        }
        
        return sumSqr;
    }    

    // todo: eliminisi pojedinacno setovanje vrednosti u rezultatu, nego napravi rowCache koji ces da system.arraycopy
    // da radi sa rows cachom u rezultatu
    /**
     * Performs dot product operation on this tensor and matrib, and stores
     * results in result tensor.
     *
     * @param matrixB
     * @param result
     * @return
     */
    // uses rowcache so it expects rows cache to be created before invocation
    public Tensor2D matMul(Tensor2D matrixB, Tensor2D result) {

        if (this.cols != matrixB.rows) {
            throw new DeepNettsException("Dimensions don't match: A.cols = " + this.cols + " B.rows = " + matrixB.rows);
        }

        if (DeepNetts.getInstance().useVectorAPI()) {
            matMulVectorized(matrixB, result);
            return result;
        }
        
        if (DeepNetts.getInstance().getMaxThreads()>1) {
            matMulMT(matrixB, result);
            return result;
        }        

        //final float[][] resultRowsCache = result.rowsCache;
        for (int rowA = 0; rowA < this.rows; rowA++) {
            final float[] row = rowsCache[rowA];
            for (int colB = 0; colB < matrixB.cols(); colB++) {
                final float[] col = matrixB.colsCache[colB];
                //  final float[] resRowCache = result.rowsCache[rowA];
                float prodSum = 0;
                for (int colA = 0; colA < cols(); colA++) {
                    prodSum += row[colA] * col[colA]; // prvo samo izmnozi a onda saberi sve elemente to moze da se vektorizuje
                    // prodSum += matrixA.get(rowA, colA) * matrixB.get(colA, colB);
                }
                //   resRowCache[colB] = prodSum;
                result.set(prodSum, rowA, colB); // ovo zameni sa result row cache i setuj row u resultu
            }
            //result.setRow(resRowCache, rowA); rowCAche setuj i
            // update values from rows cache
        }
        return result;
    }

    /**
     * Performs matrix multiplication with this 2d tensor and vector , and
     * stores results in result tensor.
     *
     * @param vectorB
     * @param result
     * @return
     */
    // uses rowcache so it expects rows cache to be created before invocation    
    public Tensor1D matMul(Tensor1D vectorB, Tensor1D result) {

        if (DeepNetts.getInstance().useVectorAPI()) {
            matMulVectorized(vectorB, result);
            return result;
        }

        //final float[][] resultRowsCache = result.rowsCache;
        final float[] productCache = new float[cols]; // use this instead prodSum below in order to enforce auto
        for (int rowA = 0; rowA < this.rows; rowA++) {
            final float[] row = rowsCache[rowA];
            final float[] vect = vectorB.values;
            //  final float[] resRowCache = result.rowsCache[rowA];
            float prodSum = 0;
            for (int colA = 0; colA < cols(); colA++) {
                prodSum += row[colA] * vect[colA]; // refactor this so it can be autovectorized
                // prodSum += matrixA.get(rowA, colA) * matrixB.get(colA, colB);
            }
            //   resRowCache[colB] = prodSum;
            result.set(prodSum, rowA); // ovo zameni sa result row cache i setuj row u resultu

            //result.setRow(resRowCache, rowA); rowCAche setuj i
            // update values from rows cache
        }
        return result;
    }

    public float[] getRowsCache(int rowIdx) {
        return rowsCache[rowIdx];
    }

    public float[] getColsCache(int colIdx) {
        return colsCache[colIdx];
    }

//    Tensor1D matMulWithAddVector(Tensor1D vector, Tensor1D result) {
//
//        if (vector.numElements() != cols) {
//            throw new IllegalArgumentException("Dimension mismatch in dotAddVector [ vector length != matrix columns ]");
//        }
//
//        float[] vec = vector.getValues();
//        float[] res = result.getValues();
//
//        int upperBound = SPECIES.loopBound(cols);
//
//        //this.createRowsCache();
//        for (int rowIdx = 0; rowIdx < this.rows; rowIdx++) {
//            FloatVector sum = FloatVector.zero(SPECIES);
//            float[] row = this.getRowsCache(rowIdx);
//
//            for (int offset = 0; offset < upperBound; offset += vecLen) {
//                FloatVector mv = FloatVector.fromArray(SPECIES, row, offset);
//                FloatVector vv = FloatVector.fromArray(SPECIES, vec, offset);
//                sum = sum.add(mv.mul(vv));
//            }
//
//            float dotProduct = sum.reduceLanes(VectorOperators.ADD);
//            for (int offset = upperBound; offset < cols; offset++) {
//                dotProduct += row[offset] * vec[offset];
//            }
//
//            res[rowIdx] = dotProduct;
//        }
//
//        result.setValues(res);
//        return result;
//    }
//
//    Tensor1D matMulFma(Tensor1D vector, Tensor1D result) {
//
//        if (vector.numElements() != cols) {
//            throw new IllegalArgumentException("Dimension mismatch in dotFma: [ vector length != matrix columns ]");
//        }
//
//        float[] vec = vector.getValues();
//        float[] res = result.getValues();
//
//        int upperBound = SPECIES.loopBound(cols);
//
//        //this.createRowsCache();
//        for (int rowIdx = 0; rowIdx < this.rows; rowIdx++) {
//            FloatVector sum = FloatVector.zero(SPECIES);
//            float[] row = this.getRowsCache(rowIdx);
//
//            for (int offset = 0; offset < upperBound; offset += vecLen) {
//                FloatVector mv = FloatVector.fromArray(SPECIES, row, offset);
//                FloatVector vv = FloatVector.fromArray(SPECIES, vec, offset);
//                sum = mv.fma(vv, sum);
//            }
//
//            float dotProduct = sum.reduceLanes(VectorOperators.ADD);
//            for (int offset = upperBound; offset < cols; offset++) {
//                dotProduct += row[offset] * vec[offset];
//            }
//
//            res[rowIdx] = dotProduct;
//        }
//
//        result.setValues(res);
//        return result;
//    }
//
//    Tensor1D matMulWithAddVectorParallel(Tensor1D vector, Tensor1D result) {
//
//        if (vector.numElements() != cols) {
//            throw new IllegalArgumentException("Dimension mismatch in dotAddVectorParallel: [ vector length != matrix columns ]");
//        }
//
//        float[] vec = vector.getValues();
//        float[] res = result.getValues();
//
//        int upperBound = SPECIES.loopBound(cols);
//
//        //this.createRowsCache();
//        IntStream.range(0, this.rows).parallel().forEach(rowIdx -> {
//            FloatVector sum = FloatVector.zero(SPECIES);
//            float[] row = this.getRowsCache(rowIdx);
//
//            for (int offset = 0; offset < upperBound; offset += vecLen) {
//                FloatVector mv = FloatVector.fromArray(SPECIES, row, offset);
//                FloatVector vv = FloatVector.fromArray(SPECIES, vec, offset);
//                sum = sum.add(mv.mul(vv));
//            }
//
//            float dotProduct = sum.reduceLanes(VectorOperators.ADD);
//            for (int offset = upperBound; offset < cols; offset++) {
//                dotProduct += row[offset] * vec[offset];
//            }
//
//            res[rowIdx] = dotProduct;
//        });
//
//        result.setValues(res);
//        return result;
//    }
//
//    Tensor1D matMulFmaParallel(Tensor1D vector, Tensor1D result) {
//
//        if (vector.numElements() != cols) {
//            throw new IllegalArgumentException("Dimension mismatch in dotFmaParallel: [ vector length != matrix columns ]");
//        }
//
//        float[] vec = vector.getValues();
//        float[] res = result.getValues();
//
//        int upperBound = SPECIES.loopBound(cols);
//
//        //this.createRowsCache();
//        IntStream.range(0, this.rows).parallel().forEach(rowIdx -> {
//            FloatVector sum = FloatVector.zero(SPECIES);
//            final float[] row = this.getRowsCache(rowIdx);
//
//            for (int offset = 0; offset < upperBound; offset += vecLen) {
//                FloatVector mv = FloatVector.fromArray(SPECIES, row, offset);
//                FloatVector vv = FloatVector.fromArray(SPECIES, vec, offset);
//                sum = mv.fma(vv, sum);
//            }
//
//            float dotProduct = sum.reduceLanes(VectorOperators.ADD);
//            for (int offset = upperBound; offset < cols; offset++) {
//                dotProduct += row[offset] * vec[offset];
//            }
//
//            res[rowIdx] = dotProduct;
//        });
//
//        result.setValues(res);
//        return result;
//    }
//
    public Tensor1D matMulVectorized(Tensor1D vector, Tensor1D result) {
        int size = this.rows();

        if (size < 224) {
            return  vectorizationImpl.matMulFma(this, vector, result);
        } else if (size >= 224 && size < 512) {
            return vectorizationImpl.matMulWithAddVector(this, vector, result);
        } else if (size >= 512 && size < 832) {
            return vectorizationImpl.matMulWithAddVectorParallel(this, vector, result);
        } else {
            return vectorizationImpl.matMulFmaParallel(this, vector, result);
        }
    }
//
//    Tensor2D matMulWithAddVector(Tensor2D matrixB, Tensor2D result) {
//
//        if (this.cols != matrixB.rows()) {
//            throw new IllegalArgumentException("Dimension missmatch in dotAddVector: [ A.cols != B.rows ]");
//        }
//
//        //this.createRowsCache();
//        //matrixB.createColsCache();
//        int upperBound = SPECIES.loopBound(cols);
//
//        for (int i = 0; i < rows; i++) {
//            float[] row = this.getRowsCache(i);
//            for (int j = 0; j < matrixB.cols; j++) {
//                float[] col = matrixB.getColsCache(j);
//                FloatVector sumVec = FloatVector.zero(SPECIES);
//                int k = 0;
//                for (; k < upperBound; k += vecLen) {
//                    FloatVector vecA = FloatVector.fromArray(SPECIES, row, k);
//                    FloatVector vecB = FloatVector.fromArray(SPECIES, col, k);
//                    sumVec = sumVec.add(vecA.mul(vecB));
//                }
//                float dot = sumVec.reduceLanes(VectorOperators.ADD);
//                for (; k < cols; k++) {
//                    dot += row[k] * col[k];
//                }
//                result.set(dot, i, j);
//            }
//        }
//
//        return result;
//    }
//
//    Tensor2D matMulFma(Tensor2D matrixB, Tensor2D result) {
//
//        if (this.cols != matrixB.rows) {
//            throw new IllegalArgumentException("Dimension missmatch in dotFma: [ A.cols != B.rows ]");
//        }
//
//        //this.createRowsCache();
//        //matrixB.createColsCache();
//        int upperBound = SPECIES.loopBound(this.cols);
//
//        for (int i = 0; i < this.rows; i++) {
//            float[] row = this.getRowsCache(i);
//            for (int j = 0; j < matrixB.cols; j++) {
//                float[] col = matrixB.getColsCache(j);
//                FloatVector sumVec = FloatVector.zero(SPECIES);
//                int k = 0;
//                for (; k < upperBound; k += vecLen) {
//                    FloatVector vecA = FloatVector.fromArray(SPECIES, row, k);
//                    FloatVector vecB = FloatVector.fromArray(SPECIES, col, k);
//                    sumVec = vecA.fma(vecB, sumVec);
//                }
//                float dot = sumVec.reduceLanes(VectorOperators.ADD);
//                for (; k < this.cols; k++) {
//                    dot += row[k] * col[k];
//                }
//                result.set(dot, i, j);
//            }
//        }
//        return result;
//    }
//
//    Tensor2D matMulWithAddVectorParallel(Tensor2D matrixB, Tensor2D result) {
//
//        if (this.cols != matrixB.rows()) {
//            throw new IllegalArgumentException("Dimension missmatch in dotAddVectorParallel: [ A.cols != B.rows ]");
//        }
//
//        int rowsA = this.rows; //broj redvova matrice A
//        int colsA = this.cols; // broj kolona matrice A = broj redova matrice B
//        int colsB = matrixB.cols; //broj kolona matrice B
//
//        //this.createRowsCache();
//        //matrixB.createColsCache();
//        int upperBound = SPECIES.loopBound(colsA);
//
//        IntStream.range(0, rowsA).parallel().forEach(i -> {
//            float[] row = this.getRowsCache(i);
//            for (int j = 0; j < colsB; j++) {
//                float[] col = matrixB.getColsCache(j);
//                FloatVector sumVec = FloatVector.zero(SPECIES);
//                int k = 0;
//                for (; k < upperBound; k += vecLen) {
//                    FloatVector vecA = FloatVector.fromArray(SPECIES, row, k);
//                    FloatVector vecB = FloatVector.fromArray(SPECIES, col, k);
//                    sumVec = sumVec.add(vecA.mul(vecB));
//                }
//                float dot = sumVec.reduceLanes(VectorOperators.ADD);
//                for (; k < colsA; k++) {
//                    dot += row[k] * col[k];
//                }
//                result.set(dot, i, j);
//            }
//        });
//        return result;
//    }
//
//    Tensor2D matMulFmaParallel(Tensor2D matrixB, Tensor2D result) {
//
//        if (this.cols != matrixB.rows()) {
//            throw new IllegalArgumentException("Dimension missmatch in dotFmaParallel: [ A.cols != B.rows ]");
//        }
//
//        //this.createRowsCache();
//        //matrixB.createColsCache();
//        int upperBound = SPECIES.loopBound(cols);
//
//        IntStream.range(0, rows).parallel().forEach(i -> {
//            float[] row = this.getRowsCache(i);
//            for (int j = 0; j < matrixB.cols; j++) {
//                float[] col = matrixB.getColsCache(j);
//                FloatVector sumVec = FloatVector.zero(SPECIES);
//                int k = 0;
//                for (; k < upperBound; k += vecLen) {
//                    FloatVector vecA = FloatVector.fromArray(SPECIES, row, k);
//                    FloatVector vecB = FloatVector.fromArray(SPECIES, col, k);
//                    sumVec = vecA.fma(vecB, sumVec);
//                }
//                float dot = sumVec.reduceLanes(VectorOperators.ADD);
//                for (; k < cols; k++) {
//                    dot += row[k] * col[k];
//                }
//                result.set(dot, i, j);
//            }
//        });
//
//        return result;
//    }

    public Tensor2D matMulVectorized(Tensor2D matrixB, Tensor2D result) {
        int size = this.rows(); // use rows * cols instead

        if (size < 128) {
            return vectorizationImpl.matMulFma(this, matrixB, result);
        } else {
            return vectorizationImpl.matMulWithAddVectorParallel(this, matrixB, result);
        }
    }

    public void outerProductAccumulate(Tensor2D inputs, Tensor2D result) {

        int D = rows();          // deltas
        int I = inputs.rows();   // inputs
        int B = inputs.cols();   // batch size

        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                float deltaVal = get(d, b);
                for (int i = 0; i < I; i++) {
                    float inputVal = inputs.get(i, b);
                    result.add(deltaVal * inputVal, d, i);
                }
            }
        }

    }

    public Tensor2D matMulMT(Tensor2D matrixB, Tensor2D result) {
        List<Callable<Tensor2D>> dotProductTasksMatrixMatrix = createMTTasks(matrixB, result);
        try {
            threadPool.getExecutorService().invokeAll(dotProductTasksMatrixMatrix);
            return result;
        } catch (InterruptedException ex) {
            throw new RuntimeException("mat Mul threading exception", ex);
        }
    }
    
    

    public void setThreadPool(DeepNettsThreadPool threadPool) {
        this.threadPool = threadPool;
    }

    private List<Callable<Tensor2D>> createMTTasks(Tensor2D matrixB, Tensor2D result) {
           int numThreads = DeepNettsThreadPool.getMaxThreadsNum(); 
           int rowsPerThread, numTasks;
           int minRowColsPerThread = 128;// 16x8, 16x16
           
           if (rows < numThreads)  { // single thread
               rowsPerThread = rows;               
               numTasks = 1;
           }
/*           if (cols*matrixB.rows < minRowColsPerThread)  {
           
           }*/
           else {
               rowsPerThread = rows / numThreads; // dodaj i slucaj da je rowsPerThread manje od thresholda
               numTasks = numThreads;                                             
           }
           // dodaj i granu da ima min redova / operacija, min broredova*broj kolona po threadu npr 32, ili min operaciaj po threadu a to je prozivod rows*cols
            //int rowsPerThread = ( rows > numThreads ? rows / numThreads : 1);
           
           List<Callable<Tensor2D>> dotProductTasksMatrixMatrix = new ArrayList<>();
           for(int t = 0; t< numTasks; t++) {
               // ako ima ostatak poslednji to mora da ihvati - najbolje i prvi ili da se raspodeli
               int startRow = rowsPerThread*t;
               int endRow = rowsPerThread*(t+1);
               if ( (rows % numTasks !=0) && (rows - endRow)<numThreads) {
                   endRow += rows - endRow; //bolje bi bilo raspodeli ti ih ravnomerno na prethodne, ali za sada radi ovako
               }
               MatrixMatrixDotProductCallable task = new MatrixMatrixDotProductCallable(matrixB, result, startRow, endRow);
               dotProductTasksMatrixMatrix.add(task);
           }            
            
            return dotProductTasksMatrixMatrix;
    }

    public Tensor1D sumByColInto(Tensor1D sumTensor) {
        // batch dim are cols
       for(int r=0; r < rows; r++) {
            float sum = 0f;           
            for (int c = 0; c < cols; c++) {
                sum += get(r, c);
            }
            sumTensor.set(sum, r);
        }
        
        return sumTensor;
    }

    
    
    
    private class MatrixMatrixDotProductCallable implements Callable<Tensor2D> {

        Tensor2D a, b, result;
        int startRow, endRow;

        // a je this u ovom slucaju
        public MatrixMatrixDotProductCallable(Tensor2D b, Tensor2D result, int startRow, int endRow) {
            this.a = Tensor2D.this;
            this.b = b;
            this.result = result;
            this.startRow = startRow;
            this.endRow = endRow;
        }


        @Override
        public Tensor2D call() throws Exception {
            for (int rowAidx = startRow; rowAidx < endRow; rowAidx++) {
                final float[] rowA = a.getRowsCache(rowAidx); //
                for (int colBidx = 0; colBidx < b.cols(); colBidx++) {
                    final float[] colB = b.getColsCache(colBidx);
                    float prodSum = 0;
                    for (int colAidx = 0; colAidx < a.cols(); colAidx++) {
                        prodSum += rowA[colAidx] * colB[colAidx];
                        // prodSum[colAidx] = rowA[colAidx] * colB[colAidx];  // ovde ce moci auto vektorizacija  a ispod ih sve saberi                       
                    }
//                    for (int i = 0; i < prodSum.length; i++) {
//                        prodSum += prodSum[i];
//                    }
                    result.set(prodSum, rowAidx, colBidx); // nemoj pojedinacno da setujes nego celu kolonu ,ali ispod ove for petlje
                }
            }
            return result;
        }

    }    

    // OVO JE ZAPRAVO MNOZENJE MATRICA STO IMAMO 
//    public void outerProductAccumulateVectorized(Tensor2D inputs, Tensor2D result) {
//        int D = rows();             // broj delta neurona
//        int I = inputs.rows();      // broj input neurona
//        int B = inputs.cols();      // batch size
//
//        float[] deltaVals = this.values;          // D x B
//        float[] inputVals = inputs.getValues();   // I x B
//        float[] resultVals = result.getValues();  // D x I
//
//        // broj kolona => batch size
//        int upperBound = SPECIES.loopBound(B);
//
//        // idemo prvo kroz inputs, da bi efikasnije napisali column major format
//        for (int i = 0; i < I; i++) {
//            for (int d = 0; d < D; d++) {
//                // vektro za ukumulaciju - sve nule
//                FloatVector accVec = FloatVector.zero(SPECIES);
//
//                int j = 0;
//                for (; j < upperBound; j += vecLen) {
//                    // delta: index = j * D + d
//                    // input: index = j * I + i
//                    FloatVector vecDelta = FloatVector.fromArray(SPECIES, deltaVals, j * D + d);
//                    FloatVector vecInput = FloatVector.fromArray(SPECIES, inputVals, j * I + i);
//
//                    accVec = accVec.add(vecDelta.mul(vecInput));
//                }
//
//                float sum = accVec.reduceLanes(VectorOperators.ADD);
//
//                // tail loop - sklarani ostatk
//                for (; j < B; j++) {
//                    float delta = deltaVals[j * D + d];
//                    float input = inputVals[j * I + i];
//                    sum += delta * input;
//                }
//
//                // column-major zapis: (i, d) → i * D + d
//                resultVals[i * D + d] += sum;
//            }
//        }
//    }
}
