package deepnetts.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class LabelProbabilities {
    private final Map<String, Float> map; // ovo je za sve moguce klase direktno sa izlaza mreze

    public LabelProbabilities(Map<String, Float> map) {
        this.map = map;
    }
          
    public Float getProbabilityOf(String label) {
        return map.get(label);
    }
    
    public List<String> getLabels() {
        return new ArrayList<>(map.keySet());
    }

    
    // bilo bi dobro i da ih sortira  
    
//    public Map<String, Float> getTop3() {
//        SortedSet<Float> values = new TreeSet<>(map.values());
//        return null;
//    }
//    
//    public Map<String, Float>  getTop5() {
//        return null;
//    }
//    
//    public Map<String, Float>  getTopK(int k) {
//        return null;
//    }
    
    @Override
    public String toString() {
        return map.toString();
    }
}
