package deepnetts.automl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * Parameter search space: a collection of parameters and methods for generating all possible combinations.
 */
public final class Parameters  {
          
    /**
     * Mapping of param name to Parameter instance which contains all possible values.
     */
    private final HashMap<String, Parameter<?>> parametersMap = new HashMap<>(); // ovde kombinatorika koristi listu vrednosti
    
    private List<ParameterCombination> searchSpace; // list of all possible parameters and corresponding values (search grid)
    
    
    public Parameters() {        
    }
    
    public <T> Parameters add(String name, T... values) {        
        Parameter<T> param = new Parameter(name, Arrays.asList(values));
        parametersMap.put(name, param);
        return this;
    }          
    
    /**
     * Adds a list of possible values for specified parameter name
     * @param <T>
     * @param name
     * @param values
     * @return 
     */
    public <T> Parameters add(String name, List<T> values) {        
        Parameter<T> param = new Parameter(name, values);
        parametersMap.put(name, param);
        return this;
    }         

    public Parameters add(Parameter param) {
        parametersMap.put(param.getName(), param);
        return this;
    }              
    
    public Parameter<?> get(String name) {
        return parametersMap.get(name);
    }      
     
    // get a list ofparameters names 
    public Set<String> names() {
        return parametersMap.keySet();
    }
    
    // get list of parameters
    public Collection<Parameter<?>> values() {
        return parametersMap.values();
    }    

    @Override
    public String toString() {
        return "Parameters{" + "parametersMap=" + parametersMap + '}';
    }
    
    
    
    /**
     * Returns parameters search space - all possible combination of all given parameter values.
     * @return 
     */
    public List<ParameterCombination> createSearchSpace() {
        searchSpace = new LinkedList<>();   // list that will contain all possible combinations of parameters in parametersMap
        for (Parameter<?> param : parametersMap.values()) { // iterate all parameters
            // pre ovoga proveri uslov za ubacivanje parametra u listu ovde?
            searchSpace = generateCombinations(param);
        }        
        removeHiddenLayersHack(); // umesto ovoga dodaj klase na kombinacije, da hiddenLayers_x dodaje samo gde treba, koliko treba
        return searchSpace;
    }
    
    
    /**
     * Recursive generation of combinations of values.
     * 
     * @param allCombinations Collections in which all parameter combinations are added
     * @paramparamvalues
     * @return 
     */
    private List<ParameterCombination> generateCombinations(Parameter<?>  param) { 
         
        // aha, u prvom prolazu kreira po jednu kombinaciju za svaku vrednost prvog parametra koji uzme
        if (searchSpace.isEmpty()) { // this is a first combination
            for (Object value : param.getValues()) { // iteriraj sve moguce vrednosti tekuceg parametra
                ParameterCombination pc = new ParameterCombination();  // kreiraj novu kombinaciju parametara
                pc.addParameter(param.getName(), value);               // dodaj u tu kombinaciju tekucu vrednost parametra  
                searchSpace.add(pc);                               // dodaj kombinaciju u listu kombinacija -- ali ovo ne bi trebalo za svaku vrednost!?
            }       
            return searchSpace;
        } else { // u svakom sledecem pozivu posle prvog vec ima prethodne kombinacije                 
            List<ParameterCombination> newCombinations = new LinkedList<>();      // nova lista kombinacija koja zamenjuje prethpodnu
            for (ParameterCombination comb : searchSpace) {                   // na svaku postojecu kombinaciju kombinaciju parametara
                for (Object value : param.getValues()) {                          // dodaj sve moguce vrednosti tekuceg parametra
                    ParameterCombination newComb = new ParameterCombination(comb.parameters);   // napravi novu kombinaciju tako sto uzmes sve parametre iz tekuce kombinacije
                    // newComb multiplicira i dopunjuje prethodne kombinacije sa tekucim parametrom i svim vrednostima
                   newComb.addParameter(param.getName(), value); 
                    newCombinations.add(newComb);                                               // i tu novu kombinaciju dodaj u listu novih kombinacija koju cu ispod da vratim kao rezultat
                }
            }

            return newCombinations; // ovo zamenjuje allCombinations i zato comb ne mora da se izbacuje
        }
    }

    public int getInt(String name) {
        return  Integer.parseInt(parametersMap.get(name).getValue().toString());
    }       

    private void removeHiddenLayersHack() {
        Iterator<ParameterCombination> pcIter = searchSpace.iterator();
        while(pcIter.hasNext()) {
            ParameterCombination pc = pcIter.next();
            List<Parameter> parameters= pc.getParameters();
            Integer hiddenLayers=0;
            for(Parameter p: parameters) {
                if (p.getName().equals("hiddenLayers")) {
                    hiddenLayers = (Integer)p.getValue();                    
                }
            }
            
            Iterator<Parameter> iter = parameters.iterator();
            while(iter.hasNext()) {
                Parameter param = iter.next();
                String paramName = param.getName();
                    if (paramName.startsWith("hiddenLayer_")) {
                            int layerIdx = Integer.parseInt(paramName.substring(paramName.indexOf("_")+1));
                            if (layerIdx > hiddenLayers) {
                                iter.remove();                               
                            }
                    }
            }        
        }
        
        for(int i = 0; i< searchSpace.size(); i++) {
            ParameterCombination pc = searchSpace.get(i);
            for(int j = i+1; j< searchSpace.size(); j++) {
                if (searchSpace.get(i)!=null && searchSpace.get(i).equals(searchSpace.get(j))) {
                    searchSpace.set(j, null);
                }
            }
        }
        
        pcIter = searchSpace.iterator();
        while(pcIter.hasNext()) {
            ParameterCombination pc = pcIter.next();
            if (pc == null) {
                pcIter.remove();
            }
        }

    }
    
    /**
     * A single combination of parameters with list of parameters (with all possible values).
     */
    public final static class ParameterCombination {
        private final List<Parameter> parameters = new ArrayList<>();

        public ParameterCombination() {
        }

        public ParameterCombination(List<Parameter> copyValues) {
            parameters.addAll(copyValues);
        }
        
        /**
         * 
         * @param <T> parameter value type
         * @param name name of this parameter
         * @param val value of this parameter
         */
        <T> void addParameter(String name, T val) {
            parameters.add(new Parameter(name, Arrays.asList(val)));
        }      

        public List<Parameter> getParameters() {
            return parameters;
        }
                        
        public Properties getAsProperties() {
            Properties prop = new Properties();
            for(Parameter p : parameters) {
                prop.setProperty(p.getName(), p.getValue().toString());
            }
            
            return prop;
        }        
                
        @Override
        public String toString() {
            return parameters.toString();
        }

        @Override
        public int hashCode() {
            int hash = 5;
            hash = 37 * hash + Objects.hashCode(this.parameters);
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
            final ParameterCombination other = (ParameterCombination) obj;
            return Objects.equals(this.parameters, other.parameters);
        }
        
        
        
    }    
    
      
    // oe stavi u neku klasu tipa  BackpropConstants. Imam nesto slicno u visrecu
    public static final String OPTIMIZER = "optimization";
    public static final String HIDDEN_NEURONS = "hiddenNeurons";
    public static final String HIDDEN_LAYERS = "hiddenLayers"; // number of hidden layers
    public static final String HIDDEN_ACTIVATION = "hiddenActivation"; // activation function for hidden layers
    public static final String OUTPUT_ACTIVATION = "outputActivation"; // activation function for hidden layers
    public static final String LEARNING_RATE = "learningRate";
    public static final String MOMENTUM = "momentum";
    public static final String MAX_ERROR = "maxError";
    public static final String STOP_ERROR = "stopError";
    public static final String MAX_EPOCHS = "maxEpochs";
    public static final String STOP_EPOCHS = "stopEpochs";
    public static final String INPUTS = "inputs";
    public static final String OUTPUTS = "outputs";             
}
