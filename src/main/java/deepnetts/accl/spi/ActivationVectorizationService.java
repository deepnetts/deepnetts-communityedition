package deepnetts.accl.spi;

import deepnetts.util.DeepNettsException;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;


public class ActivationVectorizationService {
   private static final String DEFAULT_PROVIDER = "deepnetts.vector.spi.DefaultActivationVectorizationProvider";
   private static ActivationVectorizationProvider defaultProvider;

    //All providers
    public static List<ActivationVectorizationProvider> providers() {
        List<ActivationVectorizationProvider> services = new ArrayList<>();
        ServiceLoader<ActivationVectorizationProvider> loader = ServiceLoader.load(ActivationVectorizationProvider.class);
        loader.forEach(services::add);
        return services;
    }

    //Default provider
    public static ActivationVectorizationProvider defaultProvider() {
        if (defaultProvider == null) {
            defaultProvider = provider(DEFAULT_PROVIDER);
        }
        return defaultProvider;
    }

    //provider by name
    public static ActivationVectorizationProvider provider(String providerName) {
        ServiceLoader<ActivationVectorizationProvider> loader = ServiceLoader.load(ActivationVectorizationProvider.class);
       for (ActivationVectorizationProvider provider : loader) {
           if (providerName.equals(provider.getClass().getName())) {
               return provider;
           }
       }
        throw new DeepNettsException("Accelerator Service Provider " + providerName + " not found");
    }
}
