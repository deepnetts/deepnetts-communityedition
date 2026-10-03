package deepnetts.util;

public final class Stats {
    private float min, max, mean, median, std, var, q1, q3;
    private int count;

    public float getMin() {
        return min;
    }

    public void setMin(float min) {
        this.min = min;
    }

    public float getMax() {
        return max;
    }

    public void setMax(float max) {
        this.max = max;
    }

    public float getMean() {
        return mean;
    }

    public void setMean(float mean) {
        this.mean = mean;
    }

    public float getMedian() {
        return median;
    }

    public void setMedian(float median) {
        this.median = median;
    }

    public float getStd() {
        return std;
    }

    public void setStd(float std) {
        this.std = std;
    }

    public float getVar() {
        return var;
    }

    public void setVar(float var) {
        this.var = var;
    }

    public float getQ1() {
        return q1;
    }

    public void setQ1(float q1) {
        this.q1 = q1;
    }

    public float getQ3() {
        return q3;
    }

    public void setQ3(float q3) {
        this.q3 = q3;
    }

    public float getIqr() {
        return q3 - q1;
    }
    
    public float getRange() {
        return max - min;
    }

    public int count() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }

    

    @Override
    public String toString() {
        return "Count=" + count + System.lineSeparator() +
                "Min=" + min + ", Max=" + max +", Range=" + getRange() + System.lineSeparator() +
                "Mean=" + mean + ", Median=" + median + System.lineSeparator() +
                "Std=" + std + ", Var=" + var + System.lineSeparator() +
                "Q1=" + q1 + ", Q3=" + q3 + ", IQR=" + getIqr();
    }

    
    
}
