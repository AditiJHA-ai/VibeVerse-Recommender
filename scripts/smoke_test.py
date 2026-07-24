from backend.recommender import VibeRecommender

r = VibeRecommender()
for q in ["Three Dark Crowns", "Dune Messiah", "Invisible", "No Other Name"]:
    out = r.recommend(q, target_type="all", top_n=4)
    print("===", q, "found", out["found"], "vibe", out.get("primary_vibe"))
    for rec in out["recommendations"][:4]:
        print(
            f"  {rec['similarity']:.3f} [{rec['type']}] {rec['title'][:55]} ({rec['primary_vibe']})"
        )
