package deepnetts.accl.spi;

import deepnetts.util.DeepNettsException;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;


public class TensorVectorizationService {
   private static final String DEFAULT_PROVIDER = "deepnetts.vector.spi.DefaultTensorVectorizationProvider";
   private static TensorVectorizationProvider defaultProvider;

    //All providers
    public static List<TensorVectorizationProvider> providers() {
        List<TensorVectorizationProvider> services = new ArrayList<>();
        ServiceLoader<TensorVectorizationProvider> loader = ServiceLoader.load(TensorVectorizationProvider.class);
        loader.forEach(services::add);
        return services;
    }

    //Default provider
    public static TensorVectorizationProvider defaultProvider() {
        if (defaultProvider == null) {
            defaultProvider = provider(DEFAULT_PROVIDER);
        }
        return defaultProvider;
    }

    //provider by name
    public static TensorVectorizationProvider provider(String providerName) {
        ServiceLoader<TensorVectorizationProvider> loader = ServiceLoader.load(TensorVectorizationProvider.class);
       for (TensorVectorizationProvider provider : loader) {
           if (providerName.equals(provider.getClass().getName())) {
               return provider;
           }
       }
        throw new DeepNettsException("AcceleraTensor Vectorization tor Service Provider " + providerName + " not found");
    }
}
