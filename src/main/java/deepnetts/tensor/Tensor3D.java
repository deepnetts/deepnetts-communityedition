package deepnetts.tensor;

import static deepnetts.tensor.Tensors.parseFloats;
import static deepnetts.tensor.Tensors.parseInts;
import deepnetts.util.DeepNettsException;
import deepnetts.util.Stats;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * A 3D tensor/matrix, with rows, columns and depth.
 */
public class Tensor3D extends TensorBase {

    private final int rows, cols, depth; // depth or channels or batch - umesto depth bolje channels

    /**
     * Creates a new 3D tensor with specified number of rows, columns and depth.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param depth tensor depth
     */
    public Tensor3D(int depth, int rows, int cols) { // trebalo bi depth, rows, cols - fizicki depth bi trebalo da ide prvi da bi bilo brze - ovo bi dosta toga promenilo
        super(Shape.of(depth, rows, cols));
        if (rows <= 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative or zero: " + rows);
        }
        if (cols <= 0) {
            throw new IllegalArgumentException("Number of columns cannot be negative or zero: " + cols);
        }
        if (depth <= 0) {
            throw new IllegalArgumentException("Depth cannot be negative or zero: " + depth);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;

        this.values = new float[rows * cols * depth];
    }

    /**
     * Creates a 3D tensor from specified 3D array
     *
     * @param vals 2D array of tensor values
     */
    public Tensor3D(final float[][][] vals) {
        super(Shape.of(vals.length, vals[0].length, vals[0][0].length));
        this.depth = vals.length; // ovo je channels first layout NCHW, zapravo chw
        this.rows = vals[0].length;
        this.cols = vals[0][0].length;

        this.values = new float[rows * cols * depth];

        // copy values from array
        for (int z = 0; z < depth; z++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    set(vals[z][row][col], z, row, col);
                }
            }
        }
    }

    public Tensor3D(int depth, int rows, int cols, float[] values) {
        super(Shape.of(depth, rows, cols));
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }
        if (rows * cols * depth != values.length) {
            throw new IllegalArgumentException("Number of values does not match tensor dimensions! " + values.length);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;

        this.values = values;

    }

    public final float get(final int ch, final int row, final int col) {
        final int idx = ch * rows * cols + col * rows + row;  // col major      
        // final int idx = ch * rows * cols + row * cols + col;  // row major      
        return values[idx];
    }

    public final void set(final float val, final int ch, final int row, final int col) {
        final int idx = ch * rows * cols + col * rows + row; // col major
        // final int idx = ch * rows * cols + row * cols + col;  // row major      
        values[idx] = val;
    }

    public final void add(final float value, final int z, final int row, final int col) {
        final int idx = z * rows * cols + col * rows + row;
        // final int idx = z * rows * cols + row * cols + col;  // row major      
        values[idx] += value;
    }

    public final void sub(final float val, final int z, final int row, final int col) {
        final int idx = z * rows * cols + col * rows + row;
        // final int idx = z * rows * cols + row * cols + col;  // row major      
        values[idx] -= val;
    }

    @Override
    public TensorBase copy() {
        Tensor3D newTensor = new Tensor3D(depth, rows, cols);
        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);
        return newTensor;
    }

    public final int cols() {
        return cols;
    }

    public final int rows() {
        return rows;
    }

    // napravi alias get channels, mozda tako nazovi i dimenzije tenzora
    public final int depth() {
        return depth;
    }

    // izgleda da se nigde ne koristi
    /**
     * Copies 2D channel values from this tensor into given float array buffer.
     * Used on 3D tensors.
     *
     * @param ch tensor channel to copy
     * @param buff buffer to copy values into
     */
    public final void copy2DSubChannel(final int ch, final float[] buff) {
        final int channelSize = rows * cols; // za 3D tenzor velicina kanala je rows * cols
        System.arraycopy(values, ch * channelSize, buff, 0, channelSize);
    }

    public void setRow(float[] rowData, int depthIdx, int rowIdx) {
        if (rowData.length != cols) {
            throw new RuntimeException("Row length does not match the number of tensor columns:" + rowData.length);
        }
        if (rowIdx >= rows) {
            throw new RuntimeException("Row index is greater then the  number of tensor rows:" + rowIdx);
        }
        if (depthIdx >= depth) {
            throw new RuntimeException("Depth index is greater then the  number of tensor depth:" + depthIdx);
        }

        for (int col = 0; col < rowData.length; col++) {
            set(rowData[col], depthIdx, rowIdx, col);
        }
    }

    // moze da se izbegne get nego samo od do u nizu values
    public float channelMean(int ch) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                sum += get(ch, r, c);
            }
        }

        return sum / (rows * cols);
    }

    public void divChannel(float val, int ch) {
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                final float newVal = get(ch, r, c) / val;
                set(newVal, ch, r, c);
            }
        }
    }

    public static Tensor3D fromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);

        String dimLine = lines.get(0);
        String dimStr = dimLine.substring(dimLine.lastIndexOf(":") + 1);
        int dim = Integer.parseInt(dimStr);
        String shapeStr = lines.get(1);
        shapeStr = shapeStr.substring(shapeStr.lastIndexOf(":") + 1);
        int[] shape = parseInts(shapeStr);

        if (dim == 3) {
            int depth = shape[0];
            int rows = shape[1];
            int cols = shape[2];
            Tensor3D tensor = new Tensor3D(depth, rows, cols);
            int rowIdx = 0, depthIdx = 0;
            for (int i = 3; i < lines.size(); i++) {
                float[] rowVals = parseFloats(lines.get(i));
                tensor.setRow(rowVals, depthIdx, rowIdx);
                rowIdx++;
                if (rowIdx == rows) {
                    rowIdx = 0;
                    depthIdx++;
                }
            }
            return tensor;
        }

        throw new DeepNettsException("Error reading 3d tensor from file: " + filePath);

    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(this.shape.toString());
        sb.append(" values:");
        sb.append("\n");
        for (int d = 0; d < depth; d++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    sb.append(get(d, r, c));
                    if (c < cols - 1) {
                        sb.append(",");
                    }
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
        sb.append("]");

        return sb.toString();
    }

    public void toFile(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("dims:3\n");
            writer.write(String.format("shape:%d,%d,%d\n",
                    this.depth(), this.rows(), this.cols()));
            writer.write("values:\n");

            for (int z = 0; z < this.depth(); z++) {
                for (int row = 0; row < this.rows(); row++) {
                    for (int col = 0; col < this.cols(); col++) {
                        float val = this.get(z, row, col);
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
