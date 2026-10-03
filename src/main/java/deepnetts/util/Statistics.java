package deepnetts.util;

import deepnetts.data.DataSetItem;
import deepnetts.data.TabularDataSet;
import java.util.Arrays;

/**
 * Statistical functions.
 */
public class Statistics {
        
    /* 
     * da i meod za correlation matrix
     * da radi nad listom nizom i data setom
     * ubaciti proveru normalosti tipa wilcox i DETEKCIJU OUTLIERA (da li ima vrednosti van IQR )_
     */
        
    public static Histogram histogram(final float[] values, int numBins) {
        float[] sortedValues = new float[values.length];        
        System.arraycopy(values, 0, sortedValues, 0, values.length);
        Arrays.sort(sortedValues);        

        int[] bins = new int[numBins];
        
        float min = sortedValues[0];
        if (min>0) min=0;
        float max = sortedValues[sortedValues.length-1];
        float range = max - min;
        float binWidth = range / numBins;
            
        int binIdx;
        for(int i=0; i<sortedValues.length; i++) {      
            binIdx = (int)( Math.abs(sortedValues[i] - min) / binWidth);     
            if (binIdx > bins.length-1) binIdx = bins.length-1;
            bins[binIdx]++;
        }
        
        Histogram hist = new Histogram();
        hist.setBins(bins);
        hist.setBinWidth(binWidth);
        hist.setNumBins(numBins);
        
        return hist;
    }
    
    public static Histogram histogram(final float[] values, int numBins, float min, float max) {
        float[] sortedValues = new float[values.length];        
        System.arraycopy(values, 0, sortedValues, 0, values.length);
        Arrays.sort(sortedValues);        

        int[] bins = new int[numBins];
        
        float range = max - min;
        float binWidth = range / numBins;
            
        int binIdx;
        for(int i=0; i<sortedValues.length; i++) {      
            binIdx = (int)( Math.abs(sortedValues[i] - min) / binWidth);     
            if (binIdx > bins.length-1) binIdx = bins.length-1; // TODO; doublecheck
            bins[binIdx]++;
        }
        
        Histogram hist = new Histogram();
        hist.setBins(bins);
        hist.setBinWidth(binWidth);
        hist.setNumBins(numBins);
        
        return hist;
    }    
    
    public static Histogram histogram(final float[] values) {    
        Stats stats = descriptiveStatistics(values);
        int numBins = (int)Math.sqrt(stats.count());
        return histogram(values, numBins, stats.getMin(), stats.getMax());
    }
    
    public static class Histogram {
        private int[] bins;
        private float binWidth;
        private int numBins;

        public int[] bins() {
            return bins;
        }

        public float binWidth() {
            return binWidth;
        }

        public int numBins() {
            return numBins;
        }

        public void setBins(int[] bins) {
            this.bins = bins;
        }

        public void setBinWidth(float binWidth) {
            this.binWidth = binWidth;
        }

        public void setNumBins(int numBins) {
            this.numBins = numBins;
        }
            
    }
    
//    public static Map<Column, DescriptiveStatistics> descriptiveStatistics(final DataSet<MLDataItem> values) {    
//        values.getItems().get(0).getInput();
//        // tensor je prosto niz: uraditi statistiku za niz nizova
//        return null;
//    }
    
//    public static DescriptiveStatistics descriptiveStatistics(final List<Float> values) {
//        float[] arrValues = values.toArray(new Float[values.size()]);
//        return descriptiveStatistics(arrValues);
//    }    
    
    public static Stats descriptiveStatistics(final float[] values) {
        Stats stats = new Stats();
        float min, max, mean, std, variance, median, q1,q3;
        min = values[0];
        max = values[0];
        mean = 0;
        variance=0;
              
        // calculate min, max and mean in one loop / pass
        for (int i = 0; i < values.length; i++) {
            if (values[i] < min) {
                min = values[i];
            }
            
            if (values[i] > max) {
                max = values[i];
            }            
            
            mean += values[i];
        }      
        
        mean = mean / values.length;
        
        stats.setMin(min);
        stats.setMax(max);
        stats.setMean(mean);        

        for (int i = 0; i < values.length; i++) {
            variance += Math.pow(values[i]-mean, 2);            
        }
        
        variance = variance/values.length;             
        std = (float)Math.sqrt(variance);
        
        stats.setVar(variance);
        stats.setStd(std);
        
        float[] sortedValues = new float[values.length];
        
        // median
        System.arraycopy(values, 0, sortedValues, 0, values.length);
        Arrays.sort(sortedValues);
                
        if (sortedValues.length % 2 != 0) {
            median = sortedValues[(sortedValues.length / 2)];
        } else {
            median = (sortedValues[(sortedValues.length / 2)-1] + sortedValues[(sortedValues.length / 2)]) / 2.0f;
        }

        stats.setMedian(median);
        
        // q1 q3
        int q1Idx = (int)(sortedValues.length * 0.25);
        int q3Idx = (int)(sortedValues.length * 0.75);

        stats.setQ1(sortedValues[q1Idx]);
        stats.setQ3(sortedValues[q3Idx]);
        stats.setCount(values.length);
                
        return stats;
    }
    
    
    public static float max(final float[] values) {
        float max = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
            }
        }
        return max;
    }

    public static float min(final float[] values) {
        float min = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] < min) {
                min = values[i];
            }            
        }
        return min;
    }
    
    public static Range range(final float values[]) {
        float min = values[0];
        float max = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] < min) {
                min = values[i];
            } else if (values[i] > max) {
                max = values[i];
            }
        }
        return Range.of(min, max);
    }

    public static float mean(final float[] values) {
        float mean = 0;
        for (int i = 0; i < values.length; i++) {
            mean += values[i];
        }
        mean = mean / values.length;
        return mean;
    }    
    
    public static float standardDeviation(final float[] values, final float mean) {
        float variance = 0;
        for (int i = 0; i < values.length; i++) {
            variance += Math.pow(values[i]-mean, 2);
        }
        final float std = (float)Math.sqrt(variance/(values.length-1));
        return std;
    }
    
    public static float variance(final float[] values, final float mean) {
        float variance = 0;
        for (int i = 0; i < values.length; i++) {
            variance += Math.pow(values[i]-mean, 2);
        }
        variance = variance/(values.length-1);
        return variance;
    }    
    
    /**
     * Calculates and returns Pearson correlation coefficient for the given arrays.
     * 
     * @param values1
     * @param values2
     * @return 
     */
    public static float correlation(float[] values1, float[] values2) {
        // can itbe performed over arrays of difefrent size? da li postoji ovaj uslov? radi se o karakteristikama istih tacaka
        if (values1.length != values2.length) throw new IllegalArgumentException("Arrays are not of the same size!");
                
        float r, numeratorSum=0, denominatorSum1=0, denominatorSum2=0;
        float mean1 = 0;
        float mean2 = 0;
        
        // calculate means for both arrays
        for (int i = 0; i < values1.length; i++) {
            mean1 += values1[i];
            mean2 += values2[i];
        }        
        mean1 = mean1 / values1.length;
        mean2 = mean2 / values2.length;
        
        // calculate numerator and denominator sums
        for (int i = 0; i < values1.length; i++) {
            numeratorSum += (values1[i]-mean1) * (values2[i]-mean2);
            denominatorSum1 += Math.pow(values1[i]-mean1, 2);
            denominatorSum2 += Math.pow(values2[i]-mean2, 2);
        }
        
        r = numeratorSum / (float)Math.sqrt(denominatorSum1 * denominatorSum2);
        
        return r;
    }
    
    // correlationMatrix - correlation for all posible cominations of the given attributes
    // shoul dreturn correlation matrix
    public static float[][] correlationMatrix(final float[][] values) {
        // pretpostavka je da su obe dimenzije iste, ipak proveri
        // zasto je ulaz matrica, mozda da bude data set?
        float[][] correlationMatrix = new float[values.length][values.length];
        
        // pretpostavka je da ej simetricna smanji broj izracunavanja - na dijagonali su 1
        for(int i=0; i<values.length; i++) {
            for(int j=0; j<values.length; j++) {
                correlationMatrix[i][j] = correlation(values[i], values[j]);
            }
        }
        
        return correlationMatrix;
    }
    
    public static int missingValuesCount(final float[] values) {
        int nanCount = 0;
        
        for(int i=0; i<values.length; i++) {
            if (values[i] == Float.NaN) nanCount++;
        }
        
        return nanCount;
    }

    public static int[] missingValuesCount(final float[][] values) {
        int[] nanCount = new int[values[0].length];
        
        for(int r=0; r<values.length; r++) {
            for(int c=0; c<values.length; c++) {
                if (values[r][c] == Float.NaN) nanCount[c]++;
            }
        }
        
        return nanCount;
    }
    
    public static int[] missingValuesCount(TabularDataSet<DataSetItem> dataSet) {
        int[] nanCount = new int[dataSet.getColumnNames().length]; // getColumns().size()
        
        for(DataSetItem item : dataSet) {
            final float[] inputs = item.getInput().getValues();            
            for(int c=0; c<inputs.length; c++) {
                if (inputs[c] == Float.NaN) nanCount[c]++;
            }
            
            float[] outputs = item.getTargetOutput().getValues();            
            for(int c=0; c<outputs.length; c++) {
                if (outputs[c] == Float.NaN) nanCount[inputs.length+c]++;
            }                        
        }
        
        return nanCount;
    }    
    
    // todo: contingency table: relationship between category and frequency of values
    
    //Shapiro-Wilk Test 
    // https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
    // https://www.epa.gov/sites/production/files/2015-10/documents/monitoring_appendd_1997.pdf
//    public static float checkNormalDistribution() {
//        return 0;
//    }
    
}