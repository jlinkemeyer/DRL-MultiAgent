using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MacCustomTag : MonoBehaviour
{
    private string tag;
    
    // Start is called before the first frame update
    void Start()
    {
        tag = "";
    }

    public void UpdateCustomTag(string tagName)
    {
        tag = tagName;
    }

    public bool IsTag(string tag2compare)
    {
        return tag.Equals(tag2compare);
    }
}
