using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MacGoalWasHit : MonoBehaviour
{
    private bool macGoalWasHit;
    void Start()
    {
        macGoalWasHit = false;
    }

    public void SetGoalHit(bool wasHit)
    {
        macGoalWasHit = wasHit;
    }

    public bool GetGoalHit()
    {
        return macGoalWasHit;
    }
    
}
