package deepnetts.tensor;

import deepnetts.core.DeepNetts;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import static deepnetts.tensor.Tensors.parseInts;
import deepnetts.util.DeepNettsException;
import java.io.IOException;

/**
 * One dimensional tensor - a vector or an array.
 */
public class Tensor1D extends TensorBase {

    public Tensor1D(int size) {
        super(Shape.of(size));

        if (size <= 0) {
            throw new IllegalArgumentException("Size of tensor cannot be negative or zero: " + size);
        }

        this.setShape(Shape.of(size));
        this.numDimensions = 1;
        this.values = new float[size];
    }

    /**
     * Creates a single row tensor with specified values.
     *
     * @param values values of column tensor
     */
    public Tensor1D(final float... values) {
        super(Shape.of(values.length));
        this.numDimensions = 1;
        this.values = values;
    }

    // ovaj konstruktor potencijalno izbaciti - sta ce mu shape za 1f enzor? zameni ga sa Tenzor4D tamo gde s e koristi u tf importu
    public Tensor1D(Shape shape, float[] values) {
        super(shape, values);
        this.numDimensions = 1;
    }

    /**
     * Gets value from specified index position.
     *
     * @param idx position in tensor
     * @return value at specified position
     */
    public final float get(final int idx) {
        return values[idx];
    }

    /**
     * Sets value at specified index position.
     *
     * @param idx
     * @param val
     */
    public final void set(final float val, final int idx) {
        values[idx] = val;
    }

    /**
     * Adds specified value to an element at specified index position in tensor.
     *
     * @param value value to add
     * @param idx element index
     */
    public final void add(final float value, final int idx) {
        values[idx] += value;
    }

    public final Tensor1D add(final Tensor1D toAdd) { // by row ot by col, param?

        if (values.length != toAdd.values.length) {
            throw new DeepNettsException("Lengths don't match: " + values.length + "!=" + toAdd.values.length);
        }
        if (DeepNetts.getInstance().useVectorAPI()) {
    //        vectorizationImpl.addVectorized(this, toAdd);
            return this;
        }

        for (int i = 0; i < this.numElements(); i++) {
            values[i] += toAdd.values[i];
        }

        return this;
    }


    @Override
    public TensorBase copy() {
        Tensor1D newTensor = new Tensor1D(this.values.length);
        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);
        return newTensor;
    }

    @Override
    public Object clone() {
        Tensor1D newTensor = new Tensor1D(this.values.length);
        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);
        return newTensor;
    }

    /**
     * Create and return a 1D tensor of specified values.
     *
     * @param values
     * @return
     */
    public static Tensor1D of(float[] values) {
        return new Tensor1D(values);
    }

    public static Tensor1D fromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);

        // Prva linija: dimenzija
        String dimLine = lines.get(0);
        String dimStr = dimLine.substring(dimLine.lastIndexOf(":") + 1).trim();
        int dim = Integer.parseInt(dimStr);

        // Druga linija: shape
        String shapeLine = lines.get(1);
        String shapeStr = shapeLine.substring(shapeLine.lastIndexOf(":") + 1).trim();
        int[] shape = parseInts(shapeStr); // npr: [10] za shape:10

        if (dim == 1) {
            // Telo tenzora pocinje od trece linije
            String valuesLine = lines.get(3).trim(); // preskace liniju "values:"
            String[] parts = valuesLine.split(",");
            float[] values = new float[parts.length];

            for (int i = 0; i < parts.length; i++) {
                values[i] = Float.parseFloat(parts[i].trim());
            }

            return Tensor1D.of(values);
        }

        throw new DeepNettsException("Error reading 1D tensor from file: " + filePath);
    }

    public void toFile(String filePath) throws IOException {
        int length = values.length;

        StringBuilder sb = new StringBuilder();
        sb.append("dims:1\n");
        sb.append("shape:").append(length).append("\n");
        sb.append("values:\n");

        // Svi elementi u jednom redu, odvojeni zarezima
        for (int i = 0; i < length; i++) {
            sb.append(get(i));
            if (i < length - 1) {
                sb.append(",");
            }
        }
        sb.append("\n");

        Path path = Paths.get(filePath);
        Files.write(path, sb.toString().getBytes());
    }

    public void outerProduct(Tensor1D otherTensor, Tensor2D result) {
        
        if (DeepNetts.getInstance().useVectorAPI()) {
          //  vectorizationImpl.outerProductVectorized(this, otherTensor, result);
            return;
        }
        
        float valA, valB;
        int idx;
        int rowsA = this.numElements();
        int colsB = otherTensor.numElements();

        for (int i = 0; i < rowsA; i++) {
            valA = values[i];

            for (int j = 0; j < colsB; j++) {
                valB = otherTensor.values[j];

                // COLUMN-MAJOR: idx = col * rows + row
                idx = j * rowsA + i;
                result.values[idx] = valA * valB;
            }
        }
    }
}
