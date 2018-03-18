/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P4.twitter;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.Painter;

import P4.twitter.Pair;

/**
 * SocialNetwork provides methods that operate on a social network.
 * 
 * A social network is represented by a Map<String, Set<String>> where map[A] is
 * the set of people that person A follows on Twitter, and all people are
 * represented by their Twitter usernames. Users can't follow themselves. If A
 * doesn't follow anybody, then map[A] may be the empty set, or A may not even
 * exist as a key in the map; this is true even if A is followed by other people
 * in the network. Twitter usernames are not case sensitive, so "ernie" is the
 * same as "ERNie". A username should appear at most once as a key in the map or
 * in any given map[A] set.
 * 
 * DO NOT change the method signatures and specifications of these methods, but
 * you should implement their method bodies, and you may add new public or
 * private methods or classes if you like.
 */
public class SocialNetwork {

	/**
	 * Guess who might follow whom, from evidence found in tweets.
	 * 
	 * @param tweets
	 *            a list of tweets providing the evidence, not modified by this
	 *            method.
	 * @return a social network (as defined above) in which Ernie follows Bert if
	 *         and only if there is evidence for it in the given list of tweets. One
	 *         kind of evidence that Ernie follows Bert is if Ernie
	 * @-mentions Bert in a tweet. This must be implemented. Other kinds of evidence
	 *            may be used at the implementor's discretion. All the Twitter
	 *            usernames in the returned social network must be either authors
	 *            or @-mentions in the list of tweets.
	 */
	public static Map<String, Set<String>> guessFollowsGraph(List<Tweet> tweets) {
		HashMap<String, Set<String>> map_guess = new HashMap<String,Set<String>>();
		Set<String> set_searched = new HashSet<String>();
		for(Tweet tmp_t : tweets) {
			if(!set_searched.contains(tmp_t.getAuthor())) {
				set_searched.add(tmp_t.getAuthor());
				List<Tweet> tweets_written = Filter.writtenBy(tweets,tmp_t.getAuthor());
				Set<String> set_follows = Extract.getMentionedUsers(tweets_written);
				map_guess.put(tmp_t.getAuthor(), set_follows);
			}
		}
		return map_guess;
	}
	/**
	 * find the index of name
	 * @param name and array
	 * 
	 * @return the index from 0
	 */
	public static int find_index(String str,Pair[] arr,int number) {
		for(int i=0;i<number;i++)
			if(str.toUpperCase().equals(arr[i].get_first().toUpperCase()))
				return i;
		return -1;
		
	}
	/**
	 * Find the people in a social network who have the greatest influence, in the
	 * sense that they have the most followers.
	 * 
	 * @param followsGraph
	 *            a social network (as defined above)
	 * @return a list of all distinct Twitter usernames in followsGraph, in
	 *         descending order of follower count.
	 */
	public static List<String> influencers(Map<String, Set<String>> followsGraph) {
		Set<String> set_str= followsGraph.keySet();
		int size=set_str.size()+10;
		List<String> result = new ArrayList<String>();
		Pair[] arr_count = new Pair[size];
		int counter=0;
		for(String iter : set_str) {
			Set<String> follows = followsGraph.get(iter);
			for(String iter_in : follows) {
				int index;
//				if(counter==3) {
//					for(int i=0;i<counter;i++) {
//						System.out.println(arr_count[i].get_first());
//					}
//				}
				if((index=find_index(iter_in, arr_count,counter))>=0) {
					arr_count[index].increment();
					
				}
				else {
					
					arr_count[counter++]=new Pair(iter_in, 1);
				}
			}
		}
		for(int i=0;i<counter;i++) {
			int MIN = -999;
			int index=i;
			for(int j=i;j<counter;j++) {
				if(arr_count[j].get_second()>=MIN) {
					MIN = arr_count[j].get_second();
					index=j;
				}
			}
			Pair tem;
			tem=arr_count[i];
			arr_count[i]=arr_count[index];
			arr_count[index]=tem;
		//	System.out.println(arr_count[i].get_first());
			result.add(arr_count[i].get_first());
			
		}
		return result;
			
			
			
			
		

}
	/**
	 * Get triadic closure
	 * @param followsGraph
	 * @return new followsGraph with closure
	 */
	public static void triadic(Map<String, Set<String>> followsGraph){
		for(String iter : followsGraph.keySet()) {
			Set<String> tmp = new HashSet<String>();
			Set<String> set_AtoB = followsGraph.get(iter);
			for(String iter_in : set_AtoB) {
				if(followsGraph.containsKey(iter_in)) {
					Set<String> set_BtoC = followsGraph.get(iter_in);
					for(String iter_inner : set_BtoC) {
						
						if(!set_AtoB.contains(iter_inner)) tmp.add(iter_inner);
					}
				}
				
			}
			set_AtoB.addAll(tmp);
			
		}
		
	}
	/**
	 * New influencer calculater with new discretion using triadic closure 
	 * @param  followsGraph
	 *            a social network (as defined above)
	 * @return  a list of all distinct Twitter usernames in followsGraph, in
	 *         descending order of follower count.
	 */
	public static List<String> own_influencers(Map<String, Set<String>> followsGraph) {
		triadic(followsGraph);
		return influencers(followsGraph);
	}
	
}
