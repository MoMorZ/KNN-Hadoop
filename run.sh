#! /bin/sh
rm -r -f *.class *.jar
javac KNN.java
jar cvf KNN.jar ./KNN*.class
hadoop jar KNN.jar KNN $1 $2
cat $2/*