package deepnetts.tensor;

import static deepnetts.tensor.Tensors.parseFloats;
import static deepnetts.tensor.Tensors.parseInts;
import deepnetts.util.DeepNettsException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Tensor4D extends TensorBase {

    // depth == channels, fourth= batch dim4
    private final int rows, cols, depth, fourthDim;

    public Tensor4D(int fourthDim, int depth, int rows, int cols) {
        super(Shape.of(fourthDim, depth, rows, cols));

        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }
        if (fourthDim < 0) {
            throw new IllegalArgumentException("fourthDim cannot be negative: " + fourthDim);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        this.fourthDim = fourthDim;

        this.values = new float[rows * cols * depth * fourthDim];
    }

    public Tensor4D(int fourthDim, int depth, int rows, int cols, float[] values) {
        super(Shape.of(fourthDim, depth, rows, cols));

        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }
        if (fourthDim < 0) {
            throw new IllegalArgumentException("fourthDim cannot be negative: " + fourthDim);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        this.fourthDim = fourthDim;

        this.values = values;
    }

    public Tensor4D(final float[][][][] vals) {
        super(Shape.of(vals.length, vals[0].length, vals[0][0].length, vals[0][0][0].length));

        this.fourthDim = vals.length;
        this.depth = vals[0].length;
        this.rows = vals[0][0].length;
        this.cols = vals[0][0][0].length;

        this.values = new float[rows * cols * depth * fourthDim];

        // copyFrom array
        for (int f = 0; f < fourthDim; f++) {
            for (int z = 0; z < depth; z++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        set(vals[f][z][row][col], f, z, row, col);
                    }
                }
            }
        }
    }

    public final float get(final int ch, final int inCh, final int row, final int col) {
        final int idx = ch * rows * cols * depth + inCh * rows * cols + col * rows + row;
        // final int idx = ch * rows * cols * depth + inCh * rows * cols + row * cols + col;
        return values[idx];
    }

    public final void set(final float val, final int ch, final int inCh, final int row, final int col) {
        final int idx = ch * rows * cols * depth + inCh * rows * cols + col * rows + row; // jajbrze vrti low, samo je logicki raspored dakav
//        final int idx = ch * rows * cols * depth + inCh * rows * cols + row * cols + col;
        values[idx] = val;
    }

    public final void add(final float value, final int fourth, final int z, final int row, final int col) {
        final int idx = fourth * rows * cols * depth + z * rows * cols + col * rows + row;
        // final int idx = fourth * rows * cols * depth + z * rows * cols + row * cols + col;
        values[idx] += value;
    }

    public final void sub(final float val, final int fourth, final int z, final int row, final int col) {
        final int idx = fourth * rows * cols * depth + z * rows * cols + col * rows + row;
        // final int idx = fourth * rows * cols * depth + z * rows * cols + row * cols + col;
        values[idx] -= val;
    }

    @Override
    public TensorBase copy() {
        Tensor4D newTensor = new Tensor4D(fourthDim, depth, rows, cols);
        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);
        return newTensor;
    }

    public void setValuesFromStringTransposed(String values) { // , int layoutOrd -> col, row ; nhw, nchw, nhwc
        // koji je layout u stringu a koji treba da bude u tenzoru
        // u stringu je layout row major
        // u tenzoru treba da je col major
        // po defaultu neka bude layoutOrd=0 row major, a layoutOrd=1 col major
        // problem je sto nije isti layout u tf exportu i u dn, mozda najbolje da resim export da bude kako treba
        int startPos = 0, nextPos = 0;
        int idx = 0, r = 0, c = 0, d = 0, ch = 0;
        // rows and cols are available from attributes of this tensor
        // r and c are rows and columns from string

//        while(nextPos != -1) {  // da li je ovo uslov 1728, a ne zarez
        while (idx < rows * cols * depth * fourthDim) {  // da li je ovo uslov 1728, a ne zarez
            nextPos = values.indexOf(",", startPos);
            if (nextPos == -1) {
                continue; // potencijalni bug za psolednju vrednost, radi dobro jer ima zarez pre biases
            }
            String strVal = values.substring(startPos, nextPos);
            float val = Float.parseFloat(strVal.trim());

            if (c == cols) { // ako si prosao poslednju kolonu predji u novi red
                c = 0;
                r++;
                if (r == rows) { // problem je sto ovde preskace poslednju vrednost u matrici i ne postavlja je
                    r = 0;  // idem na sledeci filter to je depth     
                    d++;
                    if (d == depth) {
                        d = 0;
                        ch++;
                    }
                }
            }

            this.set(val, ch, d, r, c); // ovde uradi transpose -@fix ovo obavezno promeni            

            startPos = nextPos + 1;
            idx++; // brojac vrednosti iz fajla            
            c++;

        }
    }

    public final int cols() {
        return cols;
    }

    public final int rows() {
        return rows;
    }

    public final int depth() {
        return depth;
    }

    public int fourthDim() {
        return fourthDim;
    }

    // tf layout weights[kernel_height, kernel_width, kernel_depth, ch] CHW layout, fourth je ch, a z je filter depth
    // subchannel je 4 dimenzija - fourth
    // kopira sub channel ch u buffer dat kao parametar
    public final void copy3DSubChannel(final int ch, final float[] buff) {
        final int channelSize = rows * cols * depth; // za 4d tenzore velicina kanala rows * cols * depth
        System.arraycopy(values, ch * channelSize, buff, 0, channelSize);
    }

    void setRow(float[] rowData, int fourthIdx, int depthIdx, int rowIdx) {
        if (rowData.length != cols) {
            throw new RuntimeException("Row length does not match the number of tensor columns:" + rowData.length);
        }
        if (rowIdx >= rows) {
            throw new RuntimeException("Row index is greater then the  number of tensor rows:" + rowIdx);
        }
        if (depthIdx >= depth) {
            throw new RuntimeException("Depth index is greater then the  number of tensor depth:" + depthIdx);
        }
        if (fourthIdx >= fourthDim) {
            throw new RuntimeException("Fourth index is greater then the fourth din of tensor:" + fourthIdx);
        }

        for (int col = 0; col < rowData.length; col++) {
            set(rowData[col], fourthIdx, depthIdx, rowIdx, col);
        }
    }

    public void setValuesAt(int batchItemIdx, float[] newValues) {
        final int dstIdx = batchItemIdx * rows * cols * depth; // za 4d tenzore velicina kanala rows * cols * depth
        System.arraycopy(newValues, 0, this.values, dstIdx, newValues.length);
    }

    public static Tensor4D fromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);

        String dimLine = lines.get(0);
        String dimStr = dimLine.substring(dimLine.lastIndexOf(":") + 1).trim();
        int dim = Integer.parseInt(dimStr);

        String shapeStr = lines.get(1);
        shapeStr = shapeStr.substring(shapeStr.lastIndexOf(":") + 1).trim();
        int[] shape = parseInts(shapeStr);

        if (dim == 4) {
            final int fourthDim = shape[0];
            final int depth = shape[1];
            final int rows = shape[2];
            final int cols = shape[3];

            Tensor4D tensor = new Tensor4D(fourthDim, depth, rows, cols);

            int lineIndex = 3; // preskačemo prve 3 linije (dims, shape, values:)

            for (int b = 0; b < fourthDim; b++) {
                for (int ch = 0; ch < depth; ch++) {
                    for (int r = 0; r < rows; r++) {

                        float[] row = parseFloats(lines.get(lineIndex));

                        tensor.setRow(row, b, ch, r);
                        lineIndex++;
                    }
                }
            }

            return tensor;
        }

        throw new DeepNettsException("Error reading 4D tensor from file: " + filePath);
    }

    public void toFile(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("dims:4\n");
            writer.write(String.format("shape:%d,%d,%d,%d\n",
                    this.fourthDim(), this.depth(), this.rows(), this.cols()));
            writer.write("values:\n");

            for (int f = 0; f < this.fourthDim(); f++) {
                for (int z = 0; z < this.depth(); z++) {
                    for (int row = 0; row < this.rows(); row++) {
                        for (int col = 0; col < this.cols(); col++) {
                            float val = this.get(f, z, row, col);
                            writer.write(Float.toString(val));
                            if (col < this.cols() - 1) {
                                writer.write(", ");
                            }
                        }
                        writer.write("\n");
                    }
                }
            }
        }
    }

}
