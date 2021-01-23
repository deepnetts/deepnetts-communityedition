# Deep Netts Community Edition

To be able to use Deep Netts in your Maven based Java project, add the following dependency into dependencies section of your *pom.xml* file:

        <dependency>
            <groupId>com.deepnetts</groupId>
            <artifactId>deepnetts-core</artifactId>
            <version>1.12</version>
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

### Building

Install the core libraries (jar files) into your local maven repository, while in the _repo root folder_, run the below:

```
mvn install
```

### Running

To run the examples in the `deepnetts-examples` folder, run the below:

```
cd deepnetts
mvn exec:java -Dexec.mainClass=deepnetts.examples.[Class name]
```

Here is a list of `Class name`s to select from:

```
IrisClassification
LinearRegression
LogisticRegression
MnistHandwrittenDigitClassification
SweedenAutoInsurance
XorExample
```

For example:

```
mvn exec:java -Dexec.mainClass=deepnetts.examples.IrisClassification
```

will result in this [IrisClassification console output](deepnetts-examples/console-outputs/IrisClassification-example.log). You can find the console outputs of the other examples [here](./deepnetts-examples/console-outputs).