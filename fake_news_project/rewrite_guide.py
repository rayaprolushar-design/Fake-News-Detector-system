def get_rewrite_tips(text: str, current_ai_score: float, flagged_words: list) -> dict:
    swaps = {
        'delve': ['look into', 'explore', 'examine'],
        'leverage': ['use', 'apply', 'make use of'],
        'crucial': ['important', 'key', 'vital'],
        'furthermore': ['also', 'and', 'plus'],
        'moreover': ['besides', 'also', 'and'],
        'however': ['but', 'still', 'yet'],
        'notably': ['especially', 'in particular'],
        'significant': ['big', 'major', 'important'],
        'straightforward': ['simple', 'easy', 'clear'],
        'subsequently': ['later', 'then', 'after that'],
        'consequently': ['so', 'as a result'],
        'utilize': ['use', 'apply'],
        'facilitate': ['help', 'make easier', 'assist'],
        'implement': ['start', 'put into action', 'do'],
        'comprehensive': ['full', 'complete', 'in-depth'],
        'pivotal': ['key', 'central', 'important'],
        'imperative': ['necessary', 'must-do', 'crucial'],
        'multifaceted': ['complex', 'varied', 'many-sided'],
        'undeniably': ['clearly', 'surely', 'true that'],
        'invaluable': ['very helpful', 'useful', 'great'],
        'vibrant': ['lively', 'bright', 'energetic'],
        'robust': ['strong', 'solid', 'tough'],
        'streamline': ['simplify', 'smooth', 'make easier'],
        'cutting-edge': ['new', 'latest', 'modern'],
        'game-changer': ['breakthrough', 'major shift'],
        'paradigm': ['example', 'model', 'pattern'],
        'synergy': ['teamwork', 'combined effort'],
        'holistic': ['overall', 'complete', 'big-picture'],
        'dive into': ['look at closely', 'discuss'],
        'it is worth': ['we should', 'consider'],
        'in conclusion': ['to wrap up', 'finally', 'to sum up'],
        'in summary': ['to sum up', 'briefly'],
        'to summarize': ['to sum up', 'in short'],
        'it is important': ['we should note', 'remember that'],
        'one must': ['you should', 'we need to'],
        'this allows': ['this lets', 'this means'],
        'this ensures': ['this makes sure', 'this helps']
    }
    
    tips = {w: swaps.get(w, ['rephrase locally']) for w in flagged_words}
    
    # Estimate new score if they fix words
    # Dropping out overused words helps drop AI score significantly.
    drop = min(len(flagged_words) * 12.5, current_ai_score) 
    estimated_new_score = round(max(current_ai_score - drop, 10.0), 1)
    
    return {
        'tips': tips,
        'estimated_new_score': estimated_new_score
    }
