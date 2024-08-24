import os
import json
import requests
import ast
import re
import time

def remove_first_line(test_string):
    if test_string.startswith("Here") and test_string.split("\n")[0].strip().endswith(":"):
        return re.sub(r'^.*\n', '', test_string, count=1)
    return test_string

def generate_text(prompt, model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.7, max_retries=3, retry_delay=5):
    headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": "You are a world-class policy analyst. Analyze the given information and generate a well-structured policy brief. Include URL citations for any facts, statistics or quotes used from the search results using the placeholder <URL_REFERENCES>.",
        "messages": [{"role": "user", "content": prompt}],
    }
    retries = 0
    while retries < max_retries:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data['content'][0]['text']
            cleaned_text = remove_first_line(response_text.strip())
            print(cleaned_text)
            tokens_used = response_data['content'][0].get('tokens', 0) # Use get() to handle missing 'tokens' key
            return cleaned_text, tokens_used
        else:
            print(f"Error calling API: {response.status_code} {response.text}")
            retries += 1
            print(f"Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(retry_delay)
    print("Max retries reached. Skipping this step.")
    return "", 0

def search_web(search_term):
    headers = {"X-API-Key": os.getenv('YOU_API_KEY')}
    params = {"query": search_term}
    return requests.get(
        f"https://api.ydc-index.io/search",
        params=params,
        headers=headers,
    ).json()

def generate_subtopic_brief(subtopic, search_data):
    all_queries = []
    total_tokens = 0
    print(f"Generating initial search queries for subtopic: {subtopic}...")
    initial_queries_prompt = f"Generate 3 search queries to gather information on the subtopic '{subtopic}' for a policy brief. Return your queries in a Python-parseable list. Return nothing but the list. Do so in one line. Start your response with [\""
    initial_queries, tokens = generate_text(initial_queries_prompt)
    total_tokens += tokens
    initial_queries = ast.literal_eval('[' + initial_queries.split('[')[1])
    print(initial_queries)
    all_queries.extend(initial_queries)

    for i in range(3):
        print(f"Performing search round {i+1} for subtopic: {subtopic}...")
        for query in initial_queries:
            search_results = search_web(query)
            search_data.append(search_results)
        print(f"Generating additional search queries for subtopic: {subtopic}...")
        additional_queries_prompt = f"Here are the search results so far for the subtopic '{subtopic}':\n\n{str(search_data)}\n\n---\n\nHere are all the search queries you have used so far for this subtopic:\n\n{str(all_queries)}\n\n---\n\nBased on the search results and previous queries, generate 3 new and unique search queries to expand the knowledge on the subtopic '{subtopic}' for a policy brief. Return your queries in a Python-parseable list. Return nothing but the list. Do so in one line. Start your response with [\""
        additional_queries, tokens = generate_text(additional_queries_prompt)
        total_tokens += tokens
        additional_queries = ast.literal_eval('[' + additional_queries.split('[')[1])
        initial_queries = additional_queries
        all_queries.extend(additional_queries)

    print(f"Generating initial policy brief for subtopic: {subtopic}...")
    brief_prompt = f"When writing your policy brief, make it incredibly detailed, thorough, specific, and well-structured. Use Markdown for formatting. Analyze the following search data and generate a comprehensive policy brief on the subtopic '{subtopic}'. Include [^1] for any facts, statistics or quotes used from the search results:\n\n{str(search_data)}"
    brief, tokens = generate_text(brief_prompt, max_tokens=4000)
    total_tokens += tokens

    for i in range(3):
        print(f"Analyzing policy brief and generating additional searches (Round {i+1}) for subtopic: {subtopic}...")
        analysis_prompt = f"Analyze the following policy brief on the subtopic '{subtopic}' and identify areas that need more detail or further information:\n\n{brief}\n\n---\n\nHere are all the search queries you have used so far for this subtopic:\n\n{str(all_queries)}\n\n---\n\nGenerate 3 new and unique search queries to fill in the gaps and provide more detail on the identified areas. Return your queries in a Python-parseable list. Return nothing but the list. Do so in one line. Start your response with [\""
        additional_queries, tokens = generate_text(analysis_prompt)
        total_tokens += tokens
        additional_queries = ast.literal_eval('[' + additional_queries.split('[')[1])
        all_queries.extend(additional_queries)
        round_search_data = []
        for query in additional_queries:
            search_results = search_web(query)
            round_search_data.append(search_results)
        print(f"Updating policy brief with additional information (Round {i+1}) for subtopic: {subtopic}...")
        update_prompt = f"Update the following policy brief on the subtopic '{subtopic}' by incorporating the new information from the additional searches. Keep all existing information and citations... only add new information and citations:\n\n{brief}\n\n---\n\nAdditional search data for this round:\n\n{str(round_search_data)}\n\n---\n\nGenerate an updated policy brief that includes the new information and provides more detail in the identified areas. Use Markdown for formatting and include [^1].."
        brief, tokens = generate_text(update_prompt, max_tokens=4000)
        total_tokens += tokens

    print(f"Generating boss feedback for subtopic: {subtopic}...")
    feedback_prompt = f"Imagine you are the boss reviewing the following policy brief on the subtopic '{subtopic}':\n\n{brief}\n\n---\n\nProvide constructive feedback on what information is missing or needs further elaboration in the policy brief. Be specific and detailed in your feedback."
    feedback, tokens = generate_text(feedback_prompt, max_tokens=1000)
    total_tokens += tokens

    print(f"Generating final round of searches based on feedback for subtopic: {subtopic}...")
    final_queries_prompt = f"Based on the following feedback from the boss regarding the subtopic '{subtopic}':\n\n{feedback}\n\n---\n\nGenerate 3 search queries to find the missing information and address the areas that need further elaboration. Return your queries in a Python-parseable list. Return nothing but the list. Do so in one line. Start your response with [\""
    final_queries, tokens = generate_text(final_queries_prompt)
    total_tokens += tokens

    # Fix for the SyntaxError
    final_queries_str = final_queries.split('[')[1].strip() 
    final_queries_str = final_queries_str.replace('\n', '')  
    final_queries = json.loads(f'[{final_queries_str}]') 

    all_queries.extend(final_queries)
    final_search_data = []
    for query in final_queries:
        search_results = search_web(query)
        final_search_data.append(search_results)

    print(f"Updating policy brief with final information for subtopic: {subtopic}...")
    final_update_prompt = f"Update the following policy brief on the subtopic '{subtopic}' by incorporating the new information from the final round of searches based on the boss's feedback. Include [^1].. for any new facts, statistics or quotes:\n\n{brief}\n\n---\n\nFinal search data:\n\n{str(final_search_data)}\n\n---\n\nGenerate the final policy brief that addresses the boss's feedback and includes the missing information. Use Markdown for formatting and include [^1].." 
    final_brief, tokens = generate_text(final_update_prompt, max_tokens=4000)
    total_tokens += tokens

    print(f"Final policy brief generated for subtopic: {subtopic}!")
    return final_brief, search_data, total_tokens

def extract_references(brief, search_data):
    references = re.findall(r'\[\^(\d+)_(\d+)\]', brief)
    formatted_references = []
    for ref, url_index in references:
        ref_num = int(ref)
        url_index = int(url_index)
        if ref_num <= len(search_data):
            if 'results' in search_data[ref_num - 1]:
                urls = [result['url'] for result in search_data[ref_num - 1]['results']]
                if url_index <= len(urls):
                    formatted_references.append(f"[^{ref}_{url_index}]: {urls[url_index - 1]}")
            else:
                formatted_references.append(f"[^{ref}_{url_index}]: Reference not found")
    return "\n".join(formatted_references)

def generate_comprehensive_brief(topic, subtopic_briefs, search_data):
    print("Generating comprehensive policy brief...")
    comprehensive_brief_prompt = f"Generate a comprehensive policy brief on the topic '{topic}' by combining the following briefs on various subtopics:\n\n{subtopic_briefs}\n\n---\n\nEnsure that the final policy brief is well-structured, coherent, and covers all the important aspects of the topic. Make sure that it includes EVERYTHING in each of the briefs, in a better structured, more info-heavy manner. Nothing -- absolutely nothing, should be left out. If you forget to include ANYTHING from any of the previous briefs, you will face the consequences. Include a table of contents. Leave nothing out. Use Markdown for formatting. Retain all <URL_REFERENCES> used throughout the brief."
    comprehensive_brief, tokens = generate_text(comprehensive_brief_prompt, model="claude-3-opus-20240229", max_tokens=4000)

    # Add custom IDs to section and subsection headings
    comprehensive_brief = re.sub(r'(#+) (.+)', lambda match: f'{match.group(1)} {match.group(2)} {{#{"-".join(match.group(2).lower().split())}}}', comprehensive_brief)

    # Add anchor links to the table of contents
    comprehensive_brief = re.sub(r'- (.+)', lambda match: f'- [{match.group(1)}](#{"-".join(match.group(1).lower().split())})', comprehensive_brief, flags=re.MULTILINE)

    print("Comprehensive policy brief generated!")

    # Flatten the search data into a single list
    flattened_search_data = [item for sublist in search_data for item in sublist]

    # Replace reference placeholders with actual URLs
    references = re.findall(r'\[\^(\d+)_(\d+)\]', comprehensive_brief)
    for ref, url_index in references:
        ref_num = int(ref)
        url_index = int(url_index)
        if ref_num <= len(flattened_search_data):
            if 'results' in flattened_search_data[ref_num - 1]:
                urls = [result['url'] for result in flattened_search_data[ref_num - 1]['results']]
                if url_index <= len(urls):
                    comprehensive_brief = comprehensive_brief.replace(f"[^{ref}_{url_index}]", f"[^{ref}_{url_index}]: {urls[url_index - 1]}")

    # Extract references and add the References section to the comprehensive brief
    references_section = extract_references(comprehensive_brief, flattened_search_data)
    comprehensive_brief += f"\n\n## References\n{references_section}"

    return comprehensive_brief, tokens

# User input
policy_topic = input("Enter the policy topic: ")
total_tokens = 0
start_time = time.time()

# Generate subtopic checklist
subtopic_checklist_prompt = f"Generate a detailed checklist of subtopics to research for the policy topic '{policy_topic}'. Return your checklist in a Python-parseable list. Return nothing but the list. Do so in one line. Maximum five sub-topics. Start your response with [\""
subtopic_checklist, tokens = generate_text(subtopic_checklist_prompt)
total_tokens += tokens
subtopic_checklist = ast.literal_eval('[' + subtopic_checklist.split('[')[1])
print(f"Subtopic Checklist: {subtopic_checklist}")

# Generate briefs for each subtopic
subtopic_briefs = []
search_data = []
for subtopic in subtopic_checklist:
    subtopic_brief, subtopic_search_data, tokens = generate_subtopic_brief(subtopic, search_data)
    total_tokens += tokens
    subtopic_briefs.append(subtopic_brief)
    search_data.extend(subtopic_search_data)

# Combine subtopic briefs into a comprehensive policy brief
comprehensive_brief, comp_tokens = generate_comprehensive_brief(policy_topic, "\n\n".join(subtopic_briefs), search_data)
total_tokens += comp_tokens

# Save the comprehensive policy brief to a file
with open("comprehensive_policy_brief.txt", "w") as file:
    file.write(comprehensive_brief)
print("Comprehensive policy brief saved as 'comprehensive_policy_brief.txt'.")

end_time = time.time()
duration = end_time - start_time

# Assuming a cost of $0.0000125 per token
total_cost = total_tokens * 0.0000125
print(f"\nTotal Tokens Used: {total_tokens}")
print(f"Total Cost: ${total_cost:.4f}")
print(f"Duration: {duration:.2f} seconds")