import os
import sys

# Load dotenv to configure twilio environment mock
from dotenv import load_dotenv
load_dotenv()

from bot import handle_message

def run_tests():
    print("=== RUNNING LOCAL WEBHOOK MULTILINGUAL DISPATCHER TESTS ===\n")
    
    # Test 1: English News Claim
    print("[TEST 1] English News Claim Analysis")
    claim = "The central bank is shutting down all digital ATMs tomorrow due to security issues."
    resp = handle_message(claim)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")
    
    # Test 2: Hindi News Claim
    print("[TEST 2] Hindi News Claim Analysis")
    claim_hi = "सरकार सभी नागरिकों को मुफ्त में 5G मोबाइल बांट रही है।"
    resp = handle_message(claim_hi)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 3: Telugu News Claim
    print("[TEST 3] Telugu News Claim Analysis")
    claim_te = "ఈ కొత్త వ్యాక్సిన్ వేసుకోవడం వల్ల ప్రజలు అనారోగ్యానికి గురవుతున్నారు."
    resp = handle_message(claim_te)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 4: Short Claim in Hindi (< 6 words)
    print("[TEST 4] Short Claim Guardrail in Hindi")
    short_hi = "मोबाइल मुफ्त में"
    resp = handle_message(short_hi)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 5: Help Welcome message check (Empty body / help word)
    print("[TEST 5] Help/Welcome Message in Hindi")
    resp = handle_message("मदद") # Triggers Hindi detection, showing Hindi welcome guide
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 6: AI Text Detector Command (/ai)
    print("[TEST 6] AI Text Detector Command (/ai)")
    ai_cmd = "/ai It is crucial to leverage comprehensive strategies that facilitate robust outcomes. Furthermore, this ensures streamlined implementation."
    resp = handle_message(ai_cmd)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 7: BMW ad question (the target bug case)
    print("[TEST 7] Brand Ad Question Filter (BMW Case)")
    ad_question = "Is the advertisement fake or the advertisement is real"
    resp = handle_message(ad_question)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 8: Pure Advertisement Content
    print("[TEST 8] Commercial Advertisement Block")
    ad_content = "BMW X1 at EMI of 29999 with 5.75% ROI and road tax benefits"
    resp = handle_message(ad_content)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 9: General Non-News Question
    print("[TEST 9] General Non-News Question Block")
    gen_question = "What is the price of gold today?"
    resp = handle_message(gen_question)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 10: /share command
    print("[TEST 10] Share/Invite Command (/share)")
    resp = handle_message("/share")
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 11: /trends command
    print("[TEST 11] Trends Command (/trends)")
    resp = handle_message("/trends")
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_tests()
