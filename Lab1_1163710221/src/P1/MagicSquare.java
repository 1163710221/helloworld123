package P1;

import java.io.*;
import java.nio.ShortBuffer;;

public class MagicSquare {
	
	public static boolean isLegalMagicSquare(String fileName) {
		File file = new File(fileName);
		StringBuffer result = new StringBuffer();
		String[][] tem = new String[1000][];
		int iterator = 0;
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));// 构造一个BufferedReader类来读取文件
			String s = null;
			while ((s = br.readLine()) != null) {

				tem[iterator++] = s.split("\t");

			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		int flag = 0;
		// 异常处理
		try {
			for (int i = 0; i < iterator; i++) {
				int lenth = tem[i].length;
				for (int j = 0; j < lenth; j++) {
					if (!tem[i][j].matches("[0-9]+"))
						throw new Exception("Input error");
				}
				if (tem[i].length != iterator)
					throw new Exception("input error");

			}
			// System.out.println("good game");
		} catch (Exception e) {
			return false;

		}

		// row first
		int sum = 0;
		for (int j = 0; j < iterator; j++) {
			sum += Integer.parseInt(tem[0][j]);
		}

		// System.out.println(sum);
		for (int i = 0; i < iterator; i++) {
			int sum_tem = 0;
			for (int j = 0; j < iterator; j++) {
				sum_tem += Integer.parseInt(tem[i][j]);
			}
			if (sum != sum_tem)
				return false;
		}
		int sum_tem = 0;
		for (int i = 0; i < iterator; i++) {
			sum_tem += Integer.parseInt(tem[i][i]);

		}
		if (sum != sum_tem)
			return false;
		sum_tem = 0;
		for (int i = 0; i < iterator; i++) {
			sum_tem += Integer.parseInt(tem[iterator - i - 1][i]);

		}
		if (sum != sum_tem)
			return false;

		for (int i = 0; i < iterator; i++) {
			sum_tem = 0;
			for (int j = 0; j < iterator; j++) {
				sum_tem += Integer.parseInt(tem[j][i]);
			}
			if (sum != sum_tem)
				return false;
		}
		return true;

	}

	public static boolean generateMagicSquare(int n) {
		int magic[][] = new int[n][n];
		int row = 0, col = n / 2, i, j, square = n * n;
		for (i = 1; i <= square; i++) {
			magic[row][col] = i;
			if (i % n == 0)
			{
				
				row++;
			}
			else {
				if (row == 0)
					row = n - 1;
				else
					row--;
				if (col == (n - 1))
					col = 0;
				else
					col++;
			}
		}
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++)
				System.out.print(magic[i][j] + "\t");
			System.out.println();
		}
		return true;
	}

	public static void main(String[] args) {
		String dir = "src/P1/txt/";
		String[] test = new String[5];
		test[0] = dir + "1.txt";
		test[1] = dir + "2.txt";
		test[2] = dir + "3.txt";
		test[3] = dir + "4.txt";
		test[4] = dir + "5.txt";
		System.out.println(isLegalMagicSquare(test[0]));
		System.out.println(isLegalMagicSquare(test[1]));
		System.out.println(isLegalMagicSquare(test[2]));
		System.out.println(isLegalMagicSquare(test[3]));
		System.out.println(isLegalMagicSquare(test[4]));
		 generateMagicSquare(5);
	}

}
