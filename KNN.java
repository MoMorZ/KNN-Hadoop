import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

public class KNN
{
    // 鸢尾花种类
    private static String[] irisType = {"Iris-setosa    ", "Iris-versicolor", "Iris-virginica "};
    // 用于存储测试集
    private static ArrayList < ArrayList < Float >> testCases = new ArrayList < ArrayList < Float >> ();
    // K 值
    private static int K = 5;

    public static int main(String[] args)
    throws IOException, ClassNotFoundException, InterruptedException
    {
        // 设置配置
        Configuration conf = new Configuration();
        // 设置 job
        Job job = Job.getInstance(conf, "knn");
        job.setJarByClass(KNN.class);
        // 设置 KNN Mapper
        job.setMapperClass(KNNMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        // 设置 KNN Reducer
        job.setReducerClass(KNNReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        // 设置 Log 到文件, DEBUG 时使用
        // PrintStream ps = new PrintStream("./log.txt");
        // System.setOut(ps);
        // 设置训练集路径
        FileInputFormat.setInputPaths(job, new Path("input"));
        // 根据第三个命令行参数设置 K 值
        K = Integer.parseUnsignedInt(args[0]);
        // 设置输出路径
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        // 提交 job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
        return 0;
    }
    // KNN Mapper 类
    public static class KNNMapper extends Mapper < LongWritable, Text, IntWritable, Text >
    {
        // getDistance 计算两个样例间的欧几里德距离, 注意 label
        private float getDistance(ArrayList < Float > testCase, ArrayList < Float > trainCase)
        {
            float ret = 0;
            for(int i = 0; i < testCase.size() - 1; i++)
            {
                ret += Math.pow(testCase.get(i) - trainCase.get(i), 2);
            }
            return(float) Math.sqrt(ret);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException
        {
            // 加载训练数据
            String[] trainCaseStr = value.toString().strip().split(",");
            if(trainCaseStr.length == 0)
            {
                return;
            }
            // 把字符串数组转成浮点数数组
            ArrayList < Float > trainCase = new ArrayList < Float > ();
            try
            {
                for(String trainCaseValue : trainCaseStr)
                {
                    trainCase.add(Float.parseFloat(trainCaseValue));
                }
            }
            catch(java.lang.NumberFormatException e) {}
            // 取 label
            Float label = trainCase.get(trainCase.size() - 1);
            // 跟每一个测试样例进行进行运算并送往 Reducer
            for(int i = 0; i < testCases.size(); i++)
            {
                ArrayList < Float > testCase = testCases.get(i);
                context.write(new IntWritable(i), new Text(String.format("%f %f", getDistance(testCase, trainCase),
                              label)));
            }
        }

        @Override
        protected void setup(Context context) throws IOException, InterruptedException
        {
            // 从 test.csv 中构造测试集
            Scanner sc = new Scanner(new File("./test.csv"));
            while(sc.hasNextLine())
            {
                String line = sc.nextLine();
                // 为空则退出
                if(line.isEmpty())
                {
                    continue;
                }
                String[] testCaseStr = line.strip().split(",");
                // System.out.println(line);
                // 构造测试样例集
                ArrayList < Float > testCase = new ArrayList < Float > ();
                for(int i = 0; i < testCaseStr.length; i++)
                {
                    // System.out.println(testCaseStr[i]);
                    testCase.add(Float.parseFloat(testCaseStr[i]));
                }
                testCases.add(testCase);
            }
            // System.out.println("size: " + testCases.size());
        }
    }
    // KNN Reducer 类
    public static class KNNReducer extends Reducer < IntWritable, Text, IntWritable, Text >
    {
        @Override
        protected void reduce(IntWritable key, Iterable < Text > values, Context context)
        throws IOException, InterruptedException
        {
            // 这里构造一个红黑树, Key 为 distance, Value 为 label
            // 因为默认比较器为从小到大, 所以每次取第一个元素即为 Key 最小的
            TreeMap < Float, Integer > treeMap = new TreeMap < > ();
            for(Text value : values)
            {
                Scanner scanner = new Scanner(value.toString());
                float distance = scanner.nextFloat();
                int label = Math.round(scanner.nextFloat());
                treeMap.put(distance, label);
            }
            // 取出距离最小的 K 个，分 label 进行次数统计
            int[] cnt = new int[3];
            for(int i = 0; i <= K; i++)
            {
                Map.Entry < Float, Integer > entry = treeMap.firstEntry();
                treeMap.remove(entry.getKey());
                cnt[entry.getValue()] ++;
            }
            // 取 label 出现次数最多的作为预测结果
            int predict = 0;
            if(cnt[1] >= cnt[0] && cnt[1] >= cnt[2]) predict = 1;
            if(cnt[2] >= cnt[0] && cnt[2] >= cnt[1]) predict = 2;

            // 结果判断
            ArrayList < Float > testCase = testCases.get(key.get());
            int ans = Math.round(testCase.get(testCase.size() - 1));
            context.write(key, new Text(String.format((ans == predict ? "√" : "×"), ans, predict)
                                        + "       Ans: " + irisType[ans] + " Predict: " + irisType[predict]));
        }
    }
}