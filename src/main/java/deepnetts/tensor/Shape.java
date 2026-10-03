package deepnetts.tensor;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Immutable class that represents Tensor shape.
 * 
 */
public final class Shape implements Serializable {
    private final int MAX_DIMENSIONS = 4;
    private final int[] dimensions;
    private final int numDimensions; // rank

    public Shape(int... dimensions) {
        if (dimensions == null) throw new IllegalArgumentException("Shape dimensions cannot be null!");
        if (dimensions.length > MAX_DIMENSIONS) throw new IllegalArgumentException("Tensor shape can have max "+MAX_DIMENSIONS+" dimesions: "+dimensions.length);

        int[] newDim = new int[dimensions.length];
        for(int i=0; i<newDim.length; i++) {
            if(i<dimensions.length) {
                if (dimensions[i]<1) throw new IllegalArgumentException("Shape dimension must be greater than zero!");
                newDim[i] = dimensions[i];
            }
        }
        this.dimensions = newDim;
        this.numDimensions = dimensions.length;
    }
    
    public int[] getDimensions() {
        return dimensions;
    }
    
    public int getDim(final int dimIdx){
        return dimensions[dimIdx];
    }
    
    public int numDimensions() {
        return numDimensions;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Shape:");
        for(int d : dimensions) {
            sb.append(d).append(",");
        }
        sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 97 * hash + Arrays.hashCode(this.dimensions);
        hash = 97 * hash + this.numDimensions;
        return hash;
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
        final Shape other = (Shape) obj;
        if (this.numDimensions != other.numDimensions) {
            return false;
        }
        return Arrays.equals(this.dimensions, other.dimensions);
    }
    
    public static Shape of(int... dimensions) {
        return new Shape(dimensions);
    }    
    
    @Override
    public Object clone() {
        Shape cloned = new Shape(this.dimensions);
        return cloned;
    }
    
    
    
    
    // @layoutfix: NCHW ? Interfejs layout koji sadrzi samo konstante? NCHW i NHWC
    // umesto atribut koristi elemente niza ali to ce jos da uspori na cpu
    public static final int ROW_IDX=0;
    public static final int COL_IDX=1;
    public static final int DEPTH_IDX=2;  // ovaj bi trebalo d abude prvi

    
    // ovo imam u cudnn TensorLayout
    // interna privatna klasa 
    // Layout  NCHW  HNWC  i unutra ove konstante
    // ili Layout nchw i hnwc
    // Layout ima atribute h n w c i onda instance
    /* ovo static final bi bio NCHW
    public static final int N_IDX=0;
    public static final int CH_IDX=1;  // ovaj bi trebalo d abude prvi    
    public static final int ROW_IDX=2;
    public static final int COL_IDX=3;

    a NHWC
    public static final int N_IDX=0;
    public static final int ROW_IDX=1;
    public static final int COL_IDX=2;
    public static final int CH_IDX=3;  // ovaj bi trebalo d abude prvi        
    
    
   */

}
