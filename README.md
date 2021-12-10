<a href="https://foojay.io/works-with-openjdk"><img align="right" src="https://github.com/foojayio/badges/raw/main/works_with_openjdk/Works-with-OpenJDK.png" width="100"></a>

# Deep Netts Community Edition

[![GPL 2.0 with CPE](https://img.shields.io/badge/License-GNU%202.0%20with%20CPE-blue.svg)](COPYING)| ![build-deepnetts-examples](https://github.com/deepnetts/deepnetts-communityedition/workflows/build-deepnetts-examples/badge.svg) | [![DeepNetts](https://img.shields.io/docker/pulls/neomatrix369/deepnetts.svg)](https://hub.docker.com/r/neomatrix369/deepnetts)

To be able to use Deep Netts in your Maven based Java project, add the following dependency into dependencies section of your *pom.xml* file:

        <dependency>
            <groupId>com.deepnetts</groupId>
            <artifactId>deepnetts-core</artifactId>
            <version>1.13.2</version>
        </dependency>
    
Learn more about Deep Netts Community adition at https://www.deepnetts.com/blog/deep-netts-community-edition

If you need 
* faster training
* higher accuracy
* beginner-friendly development tools
* advanced algorithms and features
* professional technical support 

take a look at [Deep Netts Professional Edition](https://www.deepnetts.com/product.html)

## Building and Running

### Minimum requirements

- Java 8 or above
- Maven 3.5 or above

### Building

Install the core libraries (jar files) into your local maven repository, while in the _repo root folder_, run the below:

```
mvn clean install
```

### Running

To run the examples in the `deepnetts-examples` folder, run the below:

```
cd deepnetts
mvn exec:java -Dexec.mainClass=deepnetts.examples.[Class name]

or 

cd deepnetts
java -cp target/deepnetts-examples-1.12.jar deepnetts.examples.[Class name]
```

Here is a list of `Class name`s to select from:

```
BostonHouses
Cifar10
ConvolutionalImageClassifier
CrediCardFraud
DukeDetector
IrisFlowersClassifier
LinearRegression
LoadAndUseTrainedNetwork
LogisticRegression
Mnist
QuickStart
RandomLinearDataGenerator
SpamClassifier
SweedenAutoInsurance
XorExample
```

For example enter the `deepnetts-examples` folder:

```
cd deepnetts-examples
```

and then run one of these:

```
mvn exec:java -Dexec.mainClass=deepnetts.examples.IrisFlowersClassifier
```

or 

```
java -cp target/deepnetts-examples-1.12.jar deepnetts.examples.IrisFlowersClassifier
```


will result in this [IrisFlowersClassifier console output](deepnetts-examples/console-outputs/IrisFlowersClassifier-example.log). You can find the console outputs of the other examples [here](./deepnetts-examples/console-outputs).
