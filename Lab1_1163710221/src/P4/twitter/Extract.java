/* Copyright (c) 2007-2016 MIT 6.005 course staff, all rights reserved.
 * Redistribution of original or derived work requires permission of course staff.
 */
package P4.twitter;

import java.awt.image.AreaAveragingScaleFilter;
import java.util.regex.*;
import java.time.Instant;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Extract consists of methods that extract information from a list of tweets.
 * 
 * DO NOT change the method signatures and specifications of these methods, but
 * you should implement their method bodies, and you may add new public or
 * private methods or classes if you like.
 */
public class Extract {

	/**
	 * Get the time period spanned by tweets.
	 * 
	 * @param tweets
	 *            list of tweets with distinct ids, not modified by this method.
	 * @return a minimum-length time interval that contains the timestamp of every
	 *         tweet in the list.
	 */
	public static Timespan getTimespan(List<Tweet> tweets) {
		try {
			Instant start = Instant.MAX;
			Instant end = Instant.MIN;
			for (Tweet tem : tweets) {
				if (tem.getTimestamp().isBefore(start))
					start = tem.getTimestamp();
				if (tem.getTimestamp().isAfter(end))
					end = tem.getTimestamp();

			}
			Timespan result;
			result = new Timespan(start, end);
			return result;

		} catch (IllegalArgumentException e) {
			e.printStackTrace();
		}
		return new Timespan(Instant.MAX, Instant.MIN);
	}

	/**
	 * Derive the name from String which suits spec
	 * 
	 * @param str
	 * @return the Upper case name of user else return "NULL"
	 */
	public static String getName(String str) {
		Pattern pattern = Pattern.compile("^[^a-zA-Z0-9_-]*@([a-zA-Z0-9_-]+)");
		Matcher matcher = pattern.matcher(str);
		if (matcher.find())
			return matcher.group(1);
		else
			return "NULL";

	}

	/**
	 * Get usernames mentioned in a list of tweets.
	 * 
	 * @param tweets
	 *            list of tweets with distinct ids, not modified by this method.
	 * @return the set of usernames who are mentioned in the text of the tweets. A
	 *         username-mention is "@" followed by a Twitter username (as defined by
	 *         Tweet.getAuthor()'s spec). The username-mention cannot be immediately
	 *         preceded or followed by any character valid in a Twitter username.
	 *         For this reason, an email address like bitdiddle@mit.edu does NOT
	 *         contain a mention of the username mit. Twitter usernames are
	 *         case-insensitive, and the returned set may include a username at most
	 *         once.
	 */
	public static Set<String> getMentionedUsers(List<Tweet> tweets) {
		Set<String> Users = new HashSet<String>();
		Set<String> Users_Uppercase = new HashSet<String>();
		for (Tweet tmp : tweets) {
			String[] text = tmp.getText().split(" ");
			for (int i = 0; i < text.length; i++) {
				String result = getName(text[i]);
				if (!result.equals("NULL")) {
					if (!Users_Uppercase.contains(result.toUpperCase())) {
						Users.add(result);
						Users_Uppercase.add(result.toUpperCase());
					}

				}

			}
		}
		return Users;
	}

}
