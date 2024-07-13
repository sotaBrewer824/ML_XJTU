kc = 2

if __name__ == "__main__":
    print("\n[--- Learning the parameters from Flickr ---]")
    import run_triemb_learn

    print("\n[--- Compute image representation - Holidays ---]")
    import run_embedding

    print("\n[--- Compute the scores ---]")
    import img_eval
