/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P2.turtle;

import java.util.List;

import P2.turtle.DrawableTurtle;
import P2.turtle.PenColor;
import P2.turtle.Turtle;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

public class TurtleSoup {

    /**
     * Draw a square.
     * 
     * @param turtle the turtle context
     * @param sideLength length of each side
     */

    public static void drawSquare(Turtle turtle, int sideLength) {
//        throw new RuntimeException("implement me!");
    		for(int i=0;i<1000;i++) {
    			turtle.forward(1);
    			turtle.turn(10);
    		}
    	
    	
    	

    }
    /**
     * Determine inside angles of a regular polygon.
     * 
     * There is a simple formula for calculating the inside angles of a polygon;
     * you should derive it and use it here.
     * 
     * @param sides number of sides, where sides must be > 2
     * @return angle in degrees, where 0 <= angle < 360
     */
    public static double calculateRegularPolygonAngle(int sides) {
        return 180-360/(double)sides;
    }

    /**
     * Determine number of sides given the size of interior angles of a regular polygon.
     * 
     * There is a simple formula for this; you should derive it and use it here.
     * Make sure you *properly round* the answer before you return it (see java.lang.Math).
     * HINT: it is easier if you think about the exterior angles.
     * 
     * @param angle size of interior angles in degrees, where 0 < angle < 180
     * @return the integer number of sides
     */
    public static int calculatePolygonSidesFromAngle(double angle) {
        return (int) Math.round((360.0/(180.0-angle)));
    }

    /**
     * Given the number of sides, draw a regular polygon.
     * 
     * (0,0) is the lower-left corner of the polygon; use only right-hand turns to draw.
     * 
     * @param turtle the turtle context
     * @param sides number of sides of the polygon to draw
     * @param sideLength length of each side
     */
    public static void drawRegularPolygon(Turtle turtle, int sides, int sideLength) {
        double angle=calculateRegularPolygonAngle(sides);
        for(int i=0;i<sides;i++) {
        	turtle.forward(sideLength);
        	turtle.turn(180-angle);
        }
    }

    /**
     * Given the current direction, current location, and a target location, calculate the heading
     * towards the target point.
     * 
     * The return value is the angle input to turn() that would point the turtle in the direction of
     * the target point (targetX,targetY), given that the turtle is already at the point
     * (currentX,currentY) and is facing at angle currentHeading. The angle must be expressed in
     * degrees, where 0 <= angle < 360. 
     *
     * HINT: look at http://en.wikipedia.org/wiki/Atan2 and Java's math libraries
     * 
     * @param currentHeading current direction as clockwise from north
     * @param currentX current location x-coordinate
     * @param currentY current location y-coordinate
     * @param targetX target point x-coordinate
     * @param targetY target point y-coordinate
     * @return adjustment to heading (right turn amount) to get to target point,
     *         must be 0 <= angle < 360
     */
    public static double calculateHeadingToPoint(double currentHeading, int y0, int x0,
                                                 int yt, int xt) {
    	double tan=((double)yt-(double)y0)/((double)xt-(double)x0);
		double angle=Math.toDegrees(Math.atan(tan));
			if(xt>=x0) {

				double abs_angle=angle-currentHeading;
				return abs_angle<0?360+abs_angle:abs_angle;
				
				
			}
			else {

				double abs_angle=180+angle-currentHeading;
				return abs_angle<0?360+abs_angle:abs_angle;
			}
		
	}
    

    /**
     * Given a sequence of points, calculate the heading adjustments needed to get from each point
     * to the next.
     * 
     * Assumes that the turtle starts at the first point given, facing up (i.e. 0 degrees).
     * For each subsequent point, assumes that the turtle is still facing in the direction it was
     * facing when it moved to the previous point.
     * You should use calculateHeadingToPoint() to implement this function.
     * 
     * @param xCoords list of x-coordinates (must be same length as yCoords)
     * @param yCoords list of y-coordinates (must be same length as xCoords)
     * @return list of heading adjustments between points, of size 0 if (# of points) == 0,
     *         otherwise of size (# of points) - 1
     */
    public static List<Double> calculateHeadings(List<Integer> xCoords, List<Integer> yCoords) {
    	double angle=0;
    	List<Double> list=new ArrayList<Double>();
    	for(int i=0;i<xCoords.size()-1;i++) {
        		angle=calculateHeadingToPoint(angle, xCoords.get(i),yCoords.get(i),xCoords.get(i+1) , yCoords.get(i+1));
        		list.add(angle);
        	}
    	return list;
    }

    /**
     * Draw your personal, custom art.
     * 
     * Many interesting images can be drawn using the simple implementation of a turtle.  For this
     * function, draw something interesting; the complexity can be as little or as much as you want.
     * 
     * @param turtle the turtle context
     */
    /**
     * 
     * @param turtle
     * @param y0
     * @param x0
     * @param yt
     * @param xt
     * simpler implementation
     */
	public static void GoThere(Turtle turtle,double y0,double x0,double yt,double xt) {
		double tan=(yt-y0)/(xt-x0);
		double angle=Math.toDegrees(Math.atan(tan));
		int dis=(int)Math.sqrt((yt-y0)*(yt-y0)+(xt-x0)*(xt-x0));
		if(xt>=x0) {
			turtle.turn(-angle);
			turtle.forward(dis);
			turtle.turn(angle);
		}
		else {
			turtle.turn(-angle-180);
			turtle.forward(dis);
			turtle.turn(angle+180);
		}
	}
    public static void drawPersonalArt(Turtle turtle) {
//        throw new RuntimeException("implement me!");
    	turtle.color(PenColor.BLACK);
    	turtle.turn(90);
    	File file = new File("src/P2/turtle/data.txt");
	    String[][] tem = new String[1000][];
	    int iterator=0;
	    try{
	        BufferedReader br = new BufferedReader(new FileReader(file));//构造一个BufferedReader类来读取文件
	        String s = null;		      
	        while((s = br.readLine())!=null){
	        	
	            tem[iterator++] = s.split("\t");
	            
	        }
	        br.close();    
	    }catch(Exception e){
	        e.printStackTrace();
	    }
	    int[] x=new int[1000];
	    int[] y=new int[1000]; 
	    for(int i=0;i<iterator;i++) {
	    	x[i]=Integer.parseInt(tem[i][0].trim());
	    	y[i]=Integer.parseInt(tem[i][1].trim());
	    	
	    }
	    for(int i=0;i<iterator-1;i++) {
	    	GoThere(turtle, x[i], y[i], x[i+1],y[i+1]);
	    }
    }

    /**
     * Main method.
     * 
     * This is the method that runs when you run "java TurtleSoup".
     * 
     * @param args unused
     */
    public static void main(String args[]) {
        DrawableTurtle turtle = new DrawableTurtle();
        turtle.draw();
        drawPersonalArt(turtle);
      
        
    }

}
