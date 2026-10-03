package deepnetts.net.train.opt;

import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.TrainingEvent;
import deepnetts.net.train.TrainingListener;

/**
 * https://www.coursera.org/learn/deep-neural-network/lecture/hjgIA/learning-rate-decay
 */
public final class LearningRateDecay implements TrainingListener  {
   // 1/(1+0.05*x)
    private final float initialLearningRate;
    private float decayRate = 0.05f;

    public LearningRateDecay(float initialLearningRate, float decayRate) {
        this.initialLearningRate = initialLearningRate;
        this.decayRate = decayRate;
    }
    
    // moz i da mu setujes funkciju koja prima decayRate i epoch
  
    @Override
    public void handleEvent(TrainingEvent event) {
        if (event.getType() == TrainingEvent.EPOCH_FINISHED) {
            BackpropagationTrainer trainer = event.getSource();
            final float epoch = trainer.getCurrentEpoch();
            
            final float newLr = calculateNewLearningRate(epoch);            
            trainer.updateLearningRate(newLr); // set for all layers too, and all optimizers!!!           
        }
    }

    // mozda da se ubacje u konstruktor???
    private float calculateNewLearningRate(float epoch) {
      //  float newLr = initialValue * (0.1f* (epoch/100)); // da zavisi i od drugih stvari / mozda da se prepolovi ili smanji za (10%) faktor 0.9
       // float newLr = initialValue * 0.9f; // 0.9, 0.81, 0.64, 0.36 ... 
        float newLr = initialLearningRate/(1+decayRate*epoch); // 0.9, 0.81, 0.64, 0.36 ... 
        return newLr;
    }
    
 // lr = ( 1/(1+decayRate*epoch)) * initialLearningRate
    
 // eksponencijalno  decayRate^epoch * initialLR    gdeje a <1
 // l /sqrt(epoch) * LR
 // stepenasto smanjivanje na svakih x iteracija pomnozi sa nekim koeficijentom tipa 0.5  - to sa deljenjem po modulu
// ako je ostatak nula smanji lr    
 //    
    
 // smanji lr kad pocne da stagnira - ovi mogu da budu event based, znaci kad se desi neki dogadjaj, covergence speed za 10 iteracija ~ 0
 // napraviri da moze da s evrati unazad na snimljen snapshot i odatl enastavi sa novim lr
    
    
}
