package deepnetts.net;

/**
 * Neural network architecture types.
 * Supported neural network architecture types are FEEDFORWARD and CONVOLUTIONAL,
 * and they are used for serialization and deserialization in JSON format. 
 * 
 * @see FeedForwardNetwork
 * @see ConvolutionalNetwork
 * 
 * @author Zoran Sevarac
 */
public enum NetworkType {
    FEEDFORWARD("FEEDFORWARD"),
    CONVOLUTIONAL("CONVOLUTIONAL"),
    FEEDFORWARDLM("FEEDFORWARDLM"),
    CBOW("CBOW"),
    SKIPGRAM("SKIPGRAM");
    
    private final String name;       

    private NetworkType(String s) {
        name = s;
    }    
    
    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }
    
    public static NetworkType Of(Class networkClass) {
        if (networkClass.equals(FeedForwardNetwork.class)) {
            return FEEDFORWARD;
        } else if (networkClass.equals(ConvolutionalNetwork.class)){
            return CONVOLUTIONAL;
        } else if (networkClass.equals(CBOWNetwork.class)) {
            return CBOW;
        } else if (networkClass.equals(SkipGramNetwork.class)) {
            return SKIPGRAM;
        }

       throw new RuntimeException("Unknown network type!");       
    }

    @Override
    public String toString() {
       return this.name;
    }        
}
