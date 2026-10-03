package deepnetts.accl.spi;

import deepnetts.util.DeepNettsException;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;


public class AcceleratorService {
   private static final String DEFAULT_PROVIDER = "deepnetts.cuda.spi.CudaAcceleratorProvider";
   // dodaj vector api provider, multithreaded provider 
   // opciono OneAPI, Babylon, tornado provider
   private static AcceleratorProvider defaultProvider;

    //All providers
    public static List<AcceleratorProvider> providers() {
        List<AcceleratorProvider> services = new ArrayList<>();
        ServiceLoader<AcceleratorProvider> loader = ServiceLoader.load(AcceleratorProvider.class);
        loader.forEach(services::add);
        return services;
    }

    //Default provider
    public static AcceleratorProvider defaultProvider() {
        if (defaultProvider == null) {
            defaultProvider = provider(DEFAULT_PROVIDER);
        }
        return defaultProvider;
    }

    //provider by name
    public static AcceleratorProvider provider(String providerName) {
        ServiceLoader<AcceleratorProvider> loader = ServiceLoader.load(AcceleratorProvider.class);
       for (AcceleratorProvider provider : loader) {
           if (providerName.equals(provider.getClass().getName())) {
               return provider;
           }
       }
        throw new DeepNettsException("Accelerator Service Provider " + providerName + " not found");
    }
}
