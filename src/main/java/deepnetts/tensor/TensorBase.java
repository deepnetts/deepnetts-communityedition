package deepnetts.tensor;

import deepnetts.accl.spi.AcceleratorService;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import java.io.Serializable;
import java.util.Arrays;
import java.util.function.Function;
import deepnetts.accl.AcceleratorTensorBridge;
import deepnetts.accl.spi.TensorVectorizationProvider;
import deepnetts.accl.spi.TensorVectorizationService;

/**
 * The base call for multidimensional array/tensors. It us used as a basic data
 * structure for storing inputs, outputs and internal parameters(weights) of the
 * deep learning model. Provides common arithmetic operations on tensors, and
 * interface to accelerator devices like GPU.
 */
public class TensorBase implements Tensor, Serializable {

    private static final long serialVersionUID = 9123299065043860349L;
    protected static final TensorVectorizationProvider vectorizationImpl = TensorVectorizationService.defaultProvider();

    protected Shape shape; 
    protected int numDimensions;
    // Layout

    // NCHW layout depth == channels , fourth == batch, cols == width, rows = height
    // private final int[] shape = new int[4]; // dimensions
    /*
    0: N batch idx 
    1: C channel/depth
    2: H rows
    3: W cols
    
    dodaj layout i metode koje u zavisnosti od layouta vracaju eleente iz shape niza
     */
    // dimensions = shape.length;
    // deep netts korsiti row-first order isto kao tf , to je c orderting
    // cuda kotristi col- first sto je fortran style
    //
    // u ndim indexu col je 0, row je 1, z je 2, fourth je 3 (to je stride)
    // good explanation and formulas for indexing
    // https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
    /**
     * Values stored in this tensor make it final , only input layer and tests
     * sets values
     */
    protected float values[]; // todo: use ByteBuffer instead of array in order to avoid range checking

    /**
     * Bridge for this tensor on Cuda device
     */
    private transient AcceleratorTensorBridge cudaBridge;

    // remove and replace with costructor with shape param
//    protected TensorBase() {
//
//    }
    protected TensorBase(Shape shape) {
        this.shape = shape;
        this.numDimensions = shape.numDimensions();
    }

    public TensorBase(Shape shape, final float... values) {
        // @layoutfix: ovo bi trebalo prilagoditi za NCHW layout
//        this.rows = shape.getDim(Shape.ROW_IDX);    // ovo bi bbio H odnosni 2 dimenzija
//        this.cols = shape.getDim(Shape.COL_IDX);     // ovo bi bio W odnosno 3 dimenzija
//        this.depth = shape.getDim(Shape.DEPTH_IDX);  // ovo bi bio C odnosno 1 dimenaizj
//        this.fourthDim = shape.getDim(3);            // ovo bi bio N - batch odnosno nulta dimenzija
        this.shape = shape;
        this.numDimensions = shape.numDimensions();
        this.values = values;
    }

    /**
     * Public deep copy / clone constructor.
     *
     * @param t
     */
    protected TensorBase(TensorBase t) {
        this.shape = (Shape) t.shape.clone();
        this.numDimensions = t.numDimensions;
        this.values = new float[t.values.length];

        System.arraycopy(t.values, 0, values, 0, t.values.length);
    }

    @Override
    public Object clone() {
        return new TensorBase(this);
    }

    // treba li  put metoda koja ce da ih vrati/upise nazad kad izracuna out? ili to ide u out tensor?
    // still under development dont use it!
    // pretpostavi d je dim
//    public final float getWithStride(final int[] idxs) {
//        // final int idx = idxs[3] * shape[2] * shape[1] * shape[0] + idxs[2] * shape[1] * shape[0] + idxs[1] * shape[0] + idxs[0];
//        // final int idx = fourth * rows * cols * depth             + z * rows * cols               + row * cols         + col;
//        final int idx = idxs[0] * shape[1] * shape[2] * shape[3] + idxs[1] * shape[2] * shape[3] + idxs[2] * shape[3] + idxs[3];
//        return values[idx];
//    }
    @Override
    public final float[] getValues() {
        return values;
    }

    // ovu metodu vremenom izbaciti
    public final void setValues(final float... values) {
        if (values.length != this.values.length) {
            throw new DeepNettsException("Arrays are not of same size!");
        }
        // uraditi ovde sa System.arrayCopy
        this.values = values;
    }

    public final void copyFrom(final float[] src) {
        if (src.length != this.values.length) {
            throw new DeepNettsException("Arrays are not of same size!");
        }
        System.arraycopy(src, 0, values, 0, values.length);
    }

    /**
     * Copies values from specified tensor
     * @param src 
     */
    public final void copyFrom(final TensorBase src) {
        if (!shape.equals(src.shape)) throw new DeepNettsException("Tensor shapes do not match!"); 
        
        System.arraycopy(src.getValues(), 0, values, 0, values.length);
    }

    /**
     * Rank corresponds to number of dimensions in tensor.
     *
     * @return
     */
    @Override
    public final int numDimensions() {
        return numDimensions;
    }

    protected void setShape(Shape shape) {
        this.shape = shape;
    }

    @Override
    public Shape shape() {
        return shape;
    }

    /**
     * Total number of values in tensor.
     *
     * @return
     */
    public final int numElements() {
        return values.length;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(this.shape.toString());
        sb.append(" Values: [");
        for (int i = 0; i < values.length; i++) {
            sb.append(values[i]);
            sb.append(",");
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append("]");
        return sb.toString();
    }

    public String toTsr() {
        StringBuilder sb = new StringBuilder();

        sb.append("dims:").append(numDimensions);
        sb.append("shape:").append(shape.toString());
        sb.append(" Values: [");
        for (int i = 0; i < values.length; i++) {
            sb.append(values[i]);
            sb.append(",");
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append("]");
        return sb.toString();
    }

    /**
     * Adds specified tensor t to this tensor.
     *
     * @param t tensor to add
     */
    public final TensorBase add(final TensorBase t) {
        // if (!this.shape.equals(t.shape)) throw new IllegalArgumentException("Tensors are not of the same shape"); - what about broadcasting

        for (int i = 0; i < values.length; i++) {
            values[i] += t.values[i];
        }

        return this;
    }

    public final TensorBase addInto(final TensorBase t, final TensorBase result) {
        // todo: check shape  - ne ubacuj nista dok ne budemo imali performance testove
        // if (!this.shape.equals(t.shape)) throw new IllegalArgumentException("Tensors are not of the same shape"); - what about broadcasting
        float[] resArr = result.getValues();

        for (int i = 0; i < values.length; i++) {
            resArr[i] = values[i] + t.values[i];
        }

        return this;
    }

    // not tested
    public final TensorBase add(final float val) {
        for (int i = 0; i < values.length; i++) {
            values[i] += val;
        }
        return this;
    }

    /**
     * Subtracts specified tensor t from this tensor.
     *
     * @param t tensor to subtract
     */
    public final void sub(final TensorBase t) {
        // todo: check if dimensions match
        if (!this.shape.equals(t.shape)) {
            throw new DeepNettsException("Tensors are not equal shapes");
        }

        for (int i = 0; i < values.length; i++) {
            values[i] -= t.values[i];
        }
    }

    public final TensorBase sub(final TensorBase t, final TensorBase result) {
        // todo: check if dimensions match
        if (!this.shape.equals(t.shape)) {
//            throw new DeepNettsException("Tensors are not equal shapes");
        }

        float[] resArr = result.getValues();
        for (int i = 0; i < values.length; i++) {
            resArr[i] = values[i] - t.values[i];
        }
        return result;
    }

    public final void sub(final float val) {
        for (int i = 0; i < values.length; i++) {
            values[i] -= val;
        }
    }

    /**
     * Subtracts tensor t2 from t1. The result is stored in t1.
     *
     * @param t1
     * @param t2
     */
    public final static void subInplace(final TensorBase t1, final TensorBase t2) {
        for (int i = 0; i < t1.values.length; i++) {
            t1.values[i] -= t2.values[i];
        }
    }

//    public final static float mse(final TensorBase t1, final TensorBase t2, final TensorBase result) {
//        for (int i = 0; i < t1.values.length; i++) {
//           result.values[i] = t1.values[i] - t2.values[i];
//           result.values[i] = result.values[i] * result.values[i];
//        }
//    }    
    /**
     * Divide all values in this tensor with specified value.
     *
     * @param value
     */
    public final void div(final float value) {
        for (int i = 0; i < values.length; i++) {
            values[i] /= value;
        }
    }

    // element wise division
    public final void div(final float[] divisors) {
        for (int i = 0; i < values.length; i++) {
            values[i] /= divisors[i];
        }
    }

    /**
     * Fills the entire tensor with specified value.
     *
     * @param value value used to fill tensor
     */
    public final void fill(final float value) {
        for (int i = 0; i < values.length; i++) {
            values[i] = value;
        }
    }

    /**
     * Element-wise divison with specified tensor.
     *
     * @param t
     */
    public final void div(TensorBase t) {
        // assumes both tensors are of the same shape
        for (int i = 0; i < values.length; i++) {
            this.values[i] = this.values[i] / t.values[i];
        }
    }

    // TODO: fix number of dimensions
    public TensorBase copy() {
//        TensorBase newTensor = new TensorBase(rows, cols,  depth, fourthDim);    
//        System.arraycopy(this.values, 0, newTensor.values, 0, this.values.length);                
//        return newTensor;
        //throw new RuntimeException("Not implemented");
         return new TensorBase(this);
    }

    /**
     * Applies specified function to all elements in tensor in-place.
     *
     * @param f function to apply to all elements.
     */
    public void apply(Function<Float, Float> f) {
        for (int i = 0; i < values.length; i++) {
            values[i] = f.apply(values[i]);
        }
    }

    public TensorBase apply(ActivationFunction af) {
        af.accept(this);
        return this;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final TensorBase other = (TensorBase) obj;
        if (!this.shape.equals(other.shape)) { // todo fix: implementiraj equals u shape do kraja 
            return false;
        }
        if (this.numDimensions != other.numDimensions) {
            return false;
        }
        if (!Arrays.equals(this.values, other.values)) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 41 * hash + this.shape.hashCode();
        hash = 41 * hash + this.numDimensions;
        hash = 41 * hash + Arrays.hashCode(this.values);
        return hash;
    }

    public boolean equals(TensorBase t2, float delta) {
        float[] arr2 = t2.getValues();

        for (int i = 0; i < values.length; i++) {
            if (Math.abs(values[i] - arr2[i]) > delta) {
                return false;
            }
        }
        return true;
    }

// todo: add clone using apache clone builder
    public static String valuesAsString(TensorBase[] tensors) {
        StringBuilder sb = new StringBuilder();

        for (TensorBase t : tensors) {
            sb.append(t.toString());
        }

        return sb.toString();
    }

    /**
     * Sets tensor values from CSV string.
     *
     * @param values string with comma separated tensor values
     * @throws exception if number of values in string does not match number of
     * values in tensor
     */
    public void setValuesFromString(String values) {
        int startPos = 0, nextPos = 0;
        int idx = 0;
        while (nextPos != -1) {
            nextPos = values.indexOf(",", startPos);
            if (nextPos == -1) {
                continue;
            }
            String val = values.substring(startPos, nextPos);
            this.values[idx] = Float.parseFloat(val.trim());
            idx++;
            startPos = nextPos + 1;
        }
    }

    // layoutOrd je kako treba da bude u tenzoru, a u exportu je row major zbog citljivosti
    // uradi jos i transpose
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
        while (nextPos != -1) {
            nextPos = values.indexOf(",", startPos);
            if (nextPos == -1) {
                continue; // potencijalni bug za psolednju vrednost, radi dobro jer ima zarez pre biases
            }
            String strVal = values.substring(startPos, nextPos);
            float val = Float.parseFloat(strVal.trim());

            ((Tensor2D) this).set(val, c, r); // ovde uradi transpose -@fix ovo obavezno promeni

            idx++;
            startPos = nextPos + 1;

            // rows and cols counter
            c++;
            throw new RuntimeException("This should never happen exception!");
//            if (c==rows) {// ovo nije dobro // ovo svakako prebaciti u 2d ili koji vec, vidi koji je rank pa po tome
//                c = 0;
//                r++;
//            }

        }
    }

    public float sum() {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i];
        }
        return sum;
    }

    /**
     * Returns sum of abs values of this tensor - L1 norm
     *
     * @return L1 norm
     */
    public float sumAbs() {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.abs(values[i]);
        }
        return sum;
    }

    /**
     * Returns sum of squared values of this tensor - L2 norm
     *
     * @return L2 norm
     */
    public float sumSqr() {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i] * values[i];
        }
        return sum;
    }

    /**
     * Randomize all values in tensor
     */
    public void randomize() {
        for (int i = 0; i < values.length; i++) {
            values[i] = RandomGenerator.getDefault().nextFloat();
        }
    }

    public void multiplyElementWise(TensorBase tensor2) {
        for (int i = 0; i < values.length; i++) {
            values[i] *= tensor2.values[i];
        }
    }

    /**
     * Multiplies all the values in tensor with a specified input parameter.
     *
     * @param m a multiplier for all values in tensor
     */
    public TensorBase multiply(float m) {
        for (int i = 0; i < values.length; i++) {
            values[i] *= m;
        }
        return this;
    }

    public void multiply(float mul, int idx) {
        values[idx] *= mul;
    }

    public TensorBase sqrt() {
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.sqrt(values[i]);
        }
        return this;
    }

    public TensorBase sqr() {
        for (int i = 0; i < values.length; i++) {
            values[i] = values[i] * values[i];
        }
        return this;
    }

    public float mean() {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum = sum + values[i];
        }
        return sum / (float) values.length;
    }

    public float std(final float mean) {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.pow(values[i] - mean, 2);
        }
        return (float) Math.sqrt(sum);
    }

    public void replace(float toReplace, float replaceWith) {
        for (int i = 0; i < values.length; i++) {
            if (values[i] == toReplace) {
                values[i] = replaceWith;
            }
        }
    }

    // ovu metodu izdvojiti u posebnu klasu da bi izvukao dependency napolje      
    // problem je sto imam ovaj atribut cudaTensor koji je jako zgodan, mozda napraviti neki opsti TensorBridge 
    // druga stvar je kako u layerima izvuci
    // ubaci svuda i debug mode
    // creates bridge on first access tha continue using the same instance
    // neka ima ovaj interfejs accelerator bridge i to ovde a onda razne implementacije u drugim
    public AcceleratorTensorBridge getOrCreateAccBridge() {
        if (cudaBridge == null) {
            createAcceleratorBridge();
        }
        return cudaBridge;
    }

    public AcceleratorTensorBridge createAcceleratorBridge() {
        cudaBridge = AcceleratorService.defaultProvider().createAcceleratorTensorBridge(this); // odakle da uzmem arenu
        return cudaBridge;
    }

    public AcceleratorTensorBridge getAcceleratorBridge() {
        return cudaBridge;
    }

    public void copyToGPU() {
        cudaBridge.copyToDev();
    }

    public void copyFromGPU() {
        cudaBridge.copyToHost();
    }
    
    public boolean containsNaN() {
        for(float v : this.values) {
            if (Float.isNaN(v)) return true;
        }
        
        return false;
    }

}
